#!/bin/bash

# =======================================================================
#                           --- 配置區域 ---
# =======================================================================

# --- 核心修改点: 在这里手动修改 GPU 架构 ---
# Ampere 架构 (例如 A100): 使用 "sm_80"
# Ada Lovelace 架构 (例如 RTX 4090): 使用 "sm_89"
# Hopper 架构 (例如 H100): 使用 "sm_90a"
GPU_ARCH="sm_90a"

# 编译后的可执行文件名
EXECUTABLE_NAME="test-new"
# CUDA 源代码文件名 (请确保与你本地的文件名一致)
SOURCE_FILE="test-new.cu"
# 输出数据的 CSV 文件名
OUTPUT_CSV="results_new.csv"
# Python 绘图脚本的文件名
PLOT_SCRIPT="plot-new.py"

# 每个测试配置的迭代次数
ITERATIONS=50

# =======================================================================
#               --- 在此定义要测试的参数组合 ---
# =======================================================================

# --- 修改点 1: 扩展 reads_per_thread 列表以匹配 C++ 代码 ---
# 定义要测试的 reads_per_thread 列表 (对应 C++ 模板)
# **注意**: 这里的每个值都必须在你的 .cu 文件 `launch_kernel` 函数中有对应的 case
reads_per_thread_list=(1 2 4 8 16 32 64 128 256)

# 定义要测试的 one_step 列表 (单位: Bytes)
steps=(8 32 64)
# steps=(128)

# --- 修改点 2: 扩展 stride 列表以测试更大的内存跨度 ---
# 定义要测试的 stride 列表 (单位: Bytes)
strides=(8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)


# =======================================================================
#                           --- 脚本主体 ---
# =======================================================================

# --- 1. 编译 ---
echo "Compiling ${SOURCE_FILE} for architecture ${GPU_ARCH}..."
# 我们使用 C++17 标准，这对于模板化的代码是很好的实践
nvcc -arch=${GPU_ARCH} -std=c++17 -o ${EXECUTABLE_NAME} ${SOURCE_FILE} -O3
if [ $? -ne 0 ]; then
    echo "Compilation failed. Aborting."
    exit 1
fi
echo "Compilation successful. Executable: ${EXECUTABLE_NAME}"


# --- 2. 数据采集 ---
# 创建 CSV 文件并写入新的表头
echo "reads_per_thread,step,stride,bandwidth" > ${OUTPUT_CSV}

# 最外层循环，遍历不同的 reads_per_thread
for rpt in "${reads_per_thread_list[@]}"; do
    # 中层循环，遍历不同的 one_step
    for step in "${steps[@]}"; do
        # 内层循环，遍历固定的 stride 值
        for stride in "${strides[@]}"; do
            
            # 只有当 stride >= step 时测试才有意义
            if [ ${stride} -lt ${step} ]; then
                continue
            fi

            # 检查 stride 是否为 step 的整数倍，否则跳过
            if [ $((${stride} % ${step})) -ne 0 ]; then
                continue
            fi

            echo "------------------------------------------------"
            echo "Testing: rpt=${rpt}, step=${step} bytes, stride=${stride} bytes"
            
            # 运行 benchmark 程序，并捕获输出
            # **注意**: 我们现在传入了 --reads_per_thread 参数
            output=$(./${EXECUTABLE_NAME} --step=${step} --stride=${stride} --iter=${ITERATIONS} --reads_per_thread=${rpt})
            
            # 从输出中提取带宽值
            bandwidth=$(echo "${output}" | grep "Read-Only Bandwidth" | awk '{print $3}')
            
            if [ -z "${bandwidth}" ]; then
                echo "Failed to capture bandwidth for rpt=${rpt}, step=${step}, stride=${stride}. Check C++ code output."
                # 打印错误输出以帮助调试
                echo "--- Begin Program Output ---"
                echo "${output}"
                echo "--- End Program Output ---"
                continue
            fi

            # 将结果追加到 CSV 文件
            echo "${rpt},${step},${stride},${bandwidth}" >> ${OUTPUT_CSV}
            echo "Result: Bandwidth = ${bandwidth} GB/s"
        done
    done
done

echo "------------------------------------------------"
echo "Benchmark finished. Data saved to ${OUTPUT_CSV}"
echo ""


# --- 3. 执行 Python 绘图脚本 ---
echo "--- Plotting Results ---"
if [ -f "${PLOT_SCRIPT}" ]; then
    echo "Found plotting script: ${PLOT_SCRIPT}. Running..."
    # 建议使用 python3
    python3 ${PLOT_SCRIPT}
    if [ $? -eq 0 ]; then
        echo "Plotting script executed successfully."
    else
        echo "Plotting script finished with an error."
    fi
else
    echo "Plotting script ${PLOT_SCRIPT} not found. Skipping plotting."
fi

exit 0