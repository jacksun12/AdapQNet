import json
import torch
import torch.nn as nn
import logging
from adaptqnet.models.mobilenetv2 import MixedPrecisionLayer


def load_layer_precision_options(json_path, int_only=False, model_default_options=None):
    """
    加载层精度选项，并与模型默认选项取交集
    
    Args:
        json_path: JSON文件路径
        int_only: 是否只保留INT选项
        model_default_options: 模型默认的精度选项列表，如果为None则使用标准选项
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    filtered_precisions = data['filtered_precisions']
    layer_precision_options = {}
    
    # 设置默认选项
    if model_default_options is None:
        if int_only:
            model_default_options = ["int8", "int4", "int2"]
        else:
            model_default_options = ["fp32", "fp16", "int8", "int4", "int2"]
    
    # 添加调试信息
    logger = logging.getLogger(__name__)
    logger.info(f"从JSON文件加载精度选项: {json_path}")
    logger.info(f"模型默认选项: {model_default_options}")
    logger.info(f"原始层数: {len(filtered_precisions)}")
    
    total_original_options = 0
    total_filtered_options = 0
    total_intersection_options = 0
    
    for layer, options in filtered_precisions.items():
        original_options = options.copy()
        
        # 第一步：根据int_only过滤
        if int_only:
            options = [p for p in options if p.startswith('int')]
        
        # 第二步：与模型默认选项取交集
        intersection_options = [p for p in options if p in model_default_options]
        
        # 将JSON中的层名称映射到模型中的实际层名称
        # 例如: "features.0.conv_bn_relu.precision_modules.fp32.0" -> "features.0.conv_bn_relu"
        # 或者: "features.0.conv_bn_relu.base_module.0" -> "features.0.conv_bn_relu"
        actual_layer_name = layer
        if '.precision_modules.' in layer:
            # 移除 .precision_modules.fp32.0 后缀
            actual_layer_name = layer.split('.precision_modules.')[0]
        elif '.base_module.' in layer:
            # 移除 .base_module.0 后缀
            actual_layer_name = layer.split('.base_module.')[0]
            
        layer_precision_options[actual_layer_name] = intersection_options
        
        total_original_options += len(original_options)
        total_filtered_options += len(options)
        total_intersection_options += len(intersection_options)
        
        # 记录有变化的层
        if len(original_options) != len(intersection_options):
            logger.info(f"  层 {actual_layer_name}:")
            logger.info(f"    原始选项: {original_options}")
            if int_only:
                logger.info(f"    INT过滤后: {options}")
            logger.info(f"    交集后选项: {intersection_options}")
    
    logger.info(f"精度选项统计:")
    logger.info(f"  原始总选项数: {total_original_options}")
    logger.info(f"  过滤后总选项数: {total_filtered_options}")
    logger.info(f"  交集后总选项数: {total_intersection_options}")
    logger.info(f"  最终减少比例: {((total_original_options - total_intersection_options) / total_original_options * 100):.1f}%")
    
    return layer_precision_options


def set_layer_precision_options(model, layer_precision_options):
    """设置模型中每层的精度选项"""
    logger = logging.getLogger(__name__)
    logger.info("开始设置层精度选项...")
    
    found_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            found_layers.append(name)
            logger.info(f"找到 MixedPrecisionLayer: {name}")
            
            if name in layer_precision_options:
                new_precision_options = layer_precision_options[name]
                logger.info(f"层 {name}: 更新精度选项")
                logger.info(f"  从: {module.precision_options}")
                logger.info(f"  到: {new_precision_options}")
                
                # 更新精度选项
                module.precision_options = new_precision_options
                
                # 更新alpha_pact_dict - 只保留需要的PACT参数
                new_alpha_pact_dict = {}
                for precision in new_precision_options:
                    if precision in module.alpha_pact_dict:
                        new_alpha_pact_dict[precision] = module.alpha_pact_dict[precision]
                    else:
                        # 如果新的精度选项不存在，需要创建
                        device = next(module.parameters()).device
                        new_alpha_pact_dict[precision] = nn.Parameter(torch.tensor(1.0, device=device))
                module.alpha_pact_dict = nn.ParameterDict(new_alpha_pact_dict)
                
                # 检查并调整alpha参数大小以匹配精度选项数量
                if hasattr(module, 'alpha') and module.alpha.size(0) != len(module.precision_options):
                    logger.warning(f"层 {name}: alpha大小({module.alpha.size(0)})与精度选项数量({len(module.precision_options)})不匹配")
                    logger.warning(f"精度选项: {module.precision_options}")
                    # 重新初始化alpha参数以匹配精度选项数量
                    device = next(module.parameters()).device
                    new_alpha = torch.zeros(len(module.precision_options), device=device)
                    for i, precision in enumerate(module.precision_options):
                        if precision == 'int8':
                            new_alpha[i] = 1.0
                        elif precision == 'int4':
                            new_alpha[i] = 0.5
                        elif precision == 'int2':
                            new_alpha[i] = 0.2
                        elif precision == 'fp16':
                            new_alpha[i] = 0.3
                        elif precision == 'fp32':
                            new_alpha[i] = 0.1
                        else:
                            new_alpha[i] = 0.1
                    new_alpha = torch.clamp(new_alpha, 1e-6, 10.0)
                    module.alpha = nn.Parameter(new_alpha)
                    logger.info(f"  已重新初始化alpha参数，大小: {len(module.precision_options)}")
                
                logger.info(f"  更新完成: {len(new_precision_options)} 个精度选项")
            else:
                logger.warning(f"层 {name} 不在 layer_precision_options 中，保持默认精度选项")
    
    logger.info(f"模型中找到的 MixedPrecisionLayer: {found_layers}")
    logger.info(f"layer_precision_options 中的层: {list(layer_precision_options.keys())}")
    
    # 检查是否有层没有被更新
    updated_layers = [name for name in found_layers if name in layer_precision_options]
    logger.info(f"成功更新的层: {updated_layers}")
    logger.info(f"未更新的层: {[name for name in found_layers if name not in layer_precision_options]}")


def validate_model_alpha_parameters(model):
    """验证模型中所有MixedPrecisionLayer的alpha参数大小是否正确"""
    logger = logging.getLogger(__name__)
    logger.info("正在验证模型alpha参数...")
    
    all_valid = True
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            alpha_size = module.alpha.size(0)
            options_count = len(module.precision_options)
            
            if alpha_size != options_count:
                logger.warning(f"层 {name}: alpha大小({alpha_size})与精度选项数量({options_count})不匹配")
                logger.warning(f"  精度选项: {module.precision_options}")
                all_valid = False
            else:
                logger.info(f"层 {name}: ✅ 匹配")
                logger.info(f"  精度选项: {module.precision_options}")
                logger.info(f"  Alpha大小: {alpha_size}")
                logger.info(f"  选项数量: {options_count}")
    
    if all_valid:
        logger.info("所有层的alpha参数验证通过！")
    else:
        logger.warning("发现alpha参数不匹配的层，请检查！")
    
    return all_valid


def apply_precision_config(model, config):
    """Apply precision configuration"""
    logger = logging.getLogger(__name__)
    
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer) and name in config:
            try:
                module.set_precision(config[name])
                logger.info(f"成功设置 {name} 为 {config[name]}")
            except Exception as e:
                logger.info(f"设置 {name} 精度失败: {str(e)}")
                raise e 