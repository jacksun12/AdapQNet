#!/bin/bash

echo "Starting all DNAS search tasks..."

# MobileNetV2 Mixed Precision Search (32x32)
echo "Starting MobileNetV2 Mixed Precision Search (32x32)..."
nohup env PYTHONPATH=/root python dnas_search_cifar10_mbv2.py \
    --tensor_analysis_json ../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json \
    --input_size 32 > mbv2_mixed_32x32_dnas_search.log 2>&1 &
PID1=$!
echo "MobileNetV2 Mixed Precision Search (32x32) PID: $PID1"

# MobileNetV2 Integer Precision Search (32x32)
echo "Starting MobileNetV2 Integer Precision Search (32x32)..."
nohup env PYTHONPATH=/root python dnas_search_cifar10_mbv2.py \
    --tensor_analysis_json ../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json \
    --input_size 32 \
    --int_only > mbv2_int_only_32x32_dnas_search.log 2>&1 &
PID2=$!
echo "MobileNetV2 Integer Precision Search (32x32) PID: $PID2"



echo ""
echo "All DNAS search tasks have been started!"
echo "Process PID list:"
echo "MobileNetV2 Mixed Precision (32x32): $PID1"
echo "MobileNetV2 Integer Precision (32x32): $PID2"
echo ""
echo "Use the following command to check process status:"
echo "ps aux | grep dnas_search"
echo ""
echo "Use the following command to view logs:"
echo "tail -f mbv2_mixed_32x32_dnas_search.log"
echo "tail -f mbv2_int_only_32x32_dnas_search.log"
echo ""
echo "Use the following command to stop all DNAS processes:"
echo "pkill -f dnas_search"
echo ""
echo "Use the following command to check GPU usage:"
echo "nvidia-smi"
echo ""
echo "Use the following command to check system resource usage:"
echo "htop"
echo ""
