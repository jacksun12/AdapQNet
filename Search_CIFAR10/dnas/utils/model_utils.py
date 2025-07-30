import json
import torch
import torch.nn as nn
import logging
import sys
import os

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from models.mobilenetv2 import MixedPrecisionLayer


def load_layer_precision_options(json_path, int_only=False, model_default_options=None):
    """
    Load layer precision options and take intersection with model default options
    
    Args:
        json_path: Path to JSON file
        int_only: Whether to keep only INT options
        model_default_options: Model default precision options list, if None use standard options
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    filtered_precisions = data['filtered_precisions']
    layer_precision_options = {}
    
    # Set default options
    if model_default_options is None:
        if int_only:
            model_default_options = ["int8", "int4", "int2"]
        else:
            model_default_options = ["fp32", "fp16", "int8", "int4", "int2"]
    
    # Add debug information
    logger = logging.getLogger(__name__)
    logger.info(f"Loading precision options from JSON file: {json_path}")
    logger.info(f"Model default options: {model_default_options}")
    logger.info(f"Original layer count: {len(filtered_precisions)}")
    
    total_original_options = 0
    total_filtered_options = 0
    total_intersection_options = 0
    
    for layer, options in filtered_precisions.items():
        original_options = options.copy()
        
        # Step 1: Filter based on int_only
        if int_only:
            options = [p for p in options if p.startswith('int')]
        
        # Step 2: Take intersection with model default options
        intersection_options = [p for p in options if p in model_default_options]
        
        # Map layer names from JSON to actual layer names in model
        # Example: "features.0.conv_bn_relu.precision_modules.fp32.0" -> "features.0.conv_bn_relu"
        # Or: "features.0.conv_bn_relu.base_module.0" -> "features.0.conv_bn_relu"
        actual_layer_name = layer
        if '.precision_modules.' in layer:
            # Remove .precision_modules.fp32.0 suffix
            actual_layer_name = layer.split('.precision_modules.')[0]
        elif '.base_module.' in layer:
            # Remove .base_module.0 suffix
            actual_layer_name = layer.split('.base_module.')[0]
            
        layer_precision_options[actual_layer_name] = intersection_options
        
        total_original_options += len(original_options)
        total_filtered_options += len(options)
        total_intersection_options += len(intersection_options)
        
        # Record layers with changes
        if len(original_options) != len(intersection_options):
            logger.info(f"  Layer {actual_layer_name}:")
            logger.info(f"    Original options: {original_options}")
            if int_only:
                logger.info(f"    After INT filtering: {options}")
            logger.info(f"    After intersection: {intersection_options}")
    
    logger.info(f"Precision options statistics:")
    logger.info(f"  Total original options: {total_original_options}")
    logger.info(f"  Total filtered options: {total_filtered_options}")
    logger.info(f"  Total intersection options: {total_intersection_options}")
    logger.info(f"  Final reduction ratio: {((total_original_options - total_intersection_options) / total_original_options * 100):.1f}%")
    
    return layer_precision_options


def set_layer_precision_options(model, layer_precision_options):
    """Set precision options for each layer in the model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting to set layer precision options...")
    
    found_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            found_layers.append(name)
            logger.info(f"Found MixedPrecisionLayer: {name}")
            
            if name in layer_precision_options:
                new_precision_options = layer_precision_options[name]
                logger.info(f"Layer {name}: Updating precision options")
                logger.info(f"  From: {module.precision_options}")
                logger.info(f"  To: {new_precision_options}")
                
                # Update precision options
                module.precision_options = new_precision_options
                
                # Update alpha_pact_dict - only keep needed PACT parameters
                new_alpha_pact_dict = {}
                for precision in new_precision_options:
                    if precision in module.alpha_pact_dict:
                        new_alpha_pact_dict[precision] = module.alpha_pact_dict[precision]
                    else:
                        # If new precision option doesn't exist, need to create
                        device = next(module.parameters()).device
                        new_alpha_pact_dict[precision] = nn.Parameter(torch.tensor(1.0, device=device))
                module.alpha_pact_dict = nn.ParameterDict(new_alpha_pact_dict)
                
                # Check and adjust alpha parameter size to match precision options count
                if hasattr(module, 'alpha') and module.alpha.size(0) != len(module.precision_options):
                    logger.warning(f"Layer {name}: alpha size({module.alpha.size(0)}) doesn't match precision options count({len(module.precision_options)})")
                    logger.warning(f"Precision options: {module.precision_options}")
                    # Reinitialize alpha parameters to match precision options count
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
                    logger.info(f"  Reinitialized alpha parameters, size: {len(module.precision_options)}")
                
                logger.info(f"  Update completed: {len(new_precision_options)} precision options")
            else:
                logger.warning(f"Layer {name} not in layer_precision_options, keeping default precision options")
    
    logger.info(f"MixedPrecisionLayer found in model: {found_layers}")
    logger.info(f"Layers in layer_precision_options: {list(layer_precision_options.keys())}")
    
    # Check if any layers were not updated
    updated_layers = [name for name in found_layers if name in layer_precision_options]
    logger.info(f"Successfully updated layers: {updated_layers}")
    logger.info(f"Unupdated layers: {[name for name in found_layers if name not in layer_precision_options]}")


def validate_model_alpha_parameters(model):
    """Validate that alpha parameter sizes are correct for all MixedPrecisionLayer in the model"""
    logger = logging.getLogger(__name__)
    logger.info("Validating model alpha parameters...")
    
    all_valid = True
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            alpha_size = module.alpha.size(0)
            options_count = len(module.precision_options)
            
            if alpha_size != options_count:
                logger.warning(f"Layer {name}: alpha size({alpha_size}) doesn't match precision options count({options_count})")
                logger.warning(f"  Precision options: {module.precision_options}")
                all_valid = False
            else:
                logger.info(f"Layer {name}: ✅ Match")
                logger.info(f"  Precision options: {module.precision_options}")
                logger.info(f"  Alpha size: {alpha_size}")
                logger.info(f"  Options count: {options_count}")
    
    if all_valid:
        logger.info("All layer alpha parameter validation passed!")
    else:
        logger.warning("Found layers with mismatched alpha parameters, please check!")
    
    return all_valid
