#!/bin/bash

echo "开始执行所有DNAS搜索任务..."

# MobileNetV2 混合精度搜索
echo "启动 MobileNetV2 混合精度搜索..."
nohup env PYTHONPATH=/root python dnas_search_cifar10_mbv2.py \
    --checkpoint ../pretrain/checkpoints/mobilenetv2_cifar_fp32_pretrained.pth \
    --tensor_analysis_json ../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json > mbv2_mixed_dnas_search.log 2>&1 &
PID1=$!
echo "MobileNetV2 混合精度搜索 PID: $PID1"

# MobileNetV2 整数精度搜索
echo "启动 MobileNetV2 整数精度搜索..."
nohup env PYTHONPATH=/root python dnas_search_cifar10_mbv2.py \
    --checkpoint ../pretrain/checkpoints/mobilenetv2_cifar_fp32_pretrained.pth \
    --tensor_analysis_json ../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json \
    --int_only > mbv2_int_only_dnas_search.log 2>&1 &
PID2=$!
echo "MobileNetV2 整数精度搜索 PID: $PID2"

# EfficientNet 混合精度搜索
echo "启动 EfficientNet 混合精度搜索..."
nohup env PYTHONPATH=/root python dnas_search_cifar10_efficientnet.py \
    --checkpoint ../pretrain/checkpoints/efficientnet_cifar_fp32_pretrained.pth \
    --tensor_analysis_json ../tensor_analysis_result/efficientnet_cifar10_tensor_analysis_results.json > efficientnet_mixed_dnas_search.log 2>&1 &
PID3=$!
echo "EfficientNet 混合精度搜索 PID: $PID3"

# EfficientNet 整数精度搜索
echo "启动 EfficientNet 整数精度搜索..."
nohup env PYTHONPATH=/root python dnas_search_cifar10_efficientnet.py \
    --checkpoint ../pretrain/checkpoints/efficientnet_cifar_fp32_pretrained.pth \
    --tensor_analysis_json ../tensor_analysis_result/efficientnet_cifar10_tensor_analysis_results.json \
    --int_only > efficientnet_int_only_dnas_search.log 2>&1 &
PID4=$!
echo "EfficientNet 整数精度搜索 PID: $PID4"

echo ""
echo "所有DNAS搜索任务已启动！"
echo "进程PID列表："
echo "MobileNetV2 混合精度: $PID1"
echo "MobileNetV2 整数精度: $PID2"
echo "EfficientNet 混合精度: $PID3"
echo "EfficientNet 整数精度: $PID4"
echo ""
echo "使用以下命令查看进程状态："
echo "ps aux | grep dnas_search"
echo ""
echo "使用以下命令查看日志："
echo "tail -f mbv2_mixed_dnas_search.log"
echo "tail -f mbv2_int_only_dnas_search.log"
echo "tail -f efficientnet_mixed_dnas_search.log"
echo "tail -f efficientnet_int_only_dnas_search.log"
echo ""
echo "使用以下命令停止所有DNAS进程："
echo "pkill -f dnas_search"
echo ""
echo "使用以下命令查看GPU使用情况："
echo "nvidia-smi"
echo ""
echo "使用以下命令查看系统资源使用情况："
echo "htop"
echo ""
echo "注意："
echo "- 所有任务都在后台运行"
echo "- 日志文件保存在当前目录"
echo "- 可以通过PID或pkill命令停止任务"
echo "- 建议定期检查GPU内存使用情况"