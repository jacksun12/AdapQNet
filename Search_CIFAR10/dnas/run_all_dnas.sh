#!/bin/bash

echo "开始执行所有DNAS搜索任务..."

# MobileNetV2 混合精度搜索 (32x32)
echo "启动 MobileNetV2 混合精度搜索 (32x32)..."
nohup env PYTHONPATH=/root python dnas_search_cifar10_mbv2.py \
    --tensor_analysis_json ../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json \
    --input_size 32 > mbv2_mixed_32x32_dnas_search.log 2>&1 &
PID1=$!
echo "MobileNetV2 混合精度搜索 (32x32) PID: $PID1"

# MobileNetV2 整数精度搜索 (32x32)
echo "启动 MobileNetV2 整数精度搜索 (32x32)..."
nohup env PYTHONPATH=/root python dnas_search_cifar10_mbv2.py \
    --tensor_analysis_json ../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json \
    --input_size 32 \
    --int_only > mbv2_int_only_32x32_dnas_search.log 2>&1 &
PID2=$!
echo "MobileNetV2 整数精度搜索 (32x32) PID: $PID2"

# MobileNetV2 混合精度搜索 (112x224)
echo "启动 MobileNetV2 混合精度搜索 (224x224)..."
nohup env PYTHONPATH=/root python dnas_search_cifar10_mbv2.py \
    --tensor_analysis_json ../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results_112.json \
    --input_size 112 > mbv2_mixed_112x112dnas_search.log 2>&1 &
PID3=$!
echo "MobileNetV2 混合精度搜索 (224x224) PID: $PID3"


echo ""
echo "所有DNAS搜索任务已启动！"
echo "进程PID列表："
echo "MobileNetV2 混合精度 (32x32): $PID1"
echo "MobileNetV2 整数精度 (32x32): $PID2"
echo "MobileNetV2 混合精度 (224x224): $PID3"
echo "MobileNetV2 整数精度 (224x224): $PID4"
echo ""
echo "使用以下命令查看进程状态："
echo "ps aux | grep dnas_search"
echo ""
echo "使用以下命令查看日志："
echo "tail -f mbv2_mixed_32x32_dnas_search.log"
echo "tail -f mbv2_int_only_32x32_dnas_search.log"
echo "tail -f mbv2_mixed_224x224_dnas_search.log"
echo "tail -f mbv2_int_only_224x224_dnas_search.log"
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
