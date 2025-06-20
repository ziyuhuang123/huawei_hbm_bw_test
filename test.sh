#!/bin/bash

# =======================================================================
#                           --- 配置区域 ---
# =======================================================================

# --- 核心修改点: 在这里手动修改 GPU 架构 ---
# Ampere 架构 (例如 A100): 使用 "sm_80"
# Hopper 架构 (例如 H100): 使用 "sm_90a"
GPU_ARCH="sm_90a"

# 编译后的可执行文件名
EXECUTABLE_NAME="test"
# CUDA 源代码文件名
SOURCE_FILE="test.cu"
# 输出数据的 CSV 文件名
OUTPUT_CSV="results.csv"
# 迭代次数（可以设小一点以快速看到结果，例如10）
ITERATIONS=20

# =======================================================================
#                           --- 脚本主体 ---
# =======================================================================

# --- 编译 ---
echo "Compiling ${SOURCE_FILE} for architecture ${GPU_ARCH}..."
nvcc -arch=${GPU_ARCH} -o ${EXECUTABLE_NAME} ${SOURCE_FILE} -O3 --expt-relaxed-constexpr
if [ $? -ne 0 ]; then
    echo "Compilation failed. Aborting."
    exit 1
fi
echo "Compilation successful."

# --- 数据采集 ---
# 创建 CSV 文件并写入表头
echo "step,stride,bandwidth" > ${OUTPUT_CSV}

# 定义要测试的 one_step 列表
steps=(8 32 64)
# 定义要测试的 stride 列表 (2的幂次)
strides=(8 16 32 64 128 256 512 1024 2048 4096 8192)

# 外层循环，遍历不同的 one_step
for step in "${steps[@]}"; do
    # 内层循环，遍历固定的 stride 值
    for stride in "${strides[@]}"; do
        
        # 只在 stride >= step 时执行测试
        if [ ${stride} -ge ${step} ]; then
            
            # 检查 stride 是否为 step 的整数倍
            if [ $((${stride} % ${step})) -ne 0 ]; then
                continue
            fi

            echo "------------------------------------------------"
            echo "Testing: step=${step} bytes, stride=${stride} bytes on ${GPU_ARCH}"
            
            # 运行 benchmark 程序，并捕获输出
            output=$(./${EXECUTABLE_NAME} --step=${step} --stride=${stride} --iter=${ITERATIONS})
            # 修改 grep 关键词以匹配 C++ 代码的输出
            bandwidth=$(echo "${output}" | grep "Read-Only Bandwidth" | awk '{print $3}')
            
            if [ -z "${bandwidth}" ]; then
                echo "Failed to capture bandwidth for step=${step}, stride=${stride}. Skipping."
                continue
            fi

            # 将结果追加到 CSV 文件
            echo "${step},${stride},${bandwidth}" >> ${OUTPUT_CSV}
            echo "Result: Bandwidth = ${bandwidth} GB/s"
        fi
    done
done

echo "------------------------------------------------"
echo "Benchmark finished. Data saved to ${OUTPUT_CSV}"
