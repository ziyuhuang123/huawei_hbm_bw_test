#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <stdexcept>

// 用于错误检查的宏
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// =================================================================================
// 1. 模板化的 Kernel
// =================================================================================

// 使用模板定义一个可以处理任意字节大小的辅助结构体
// CUDA 内置了 1, 2, 4, 8, 16 字节的向量类型
// 我们可以自己组合来创建更大的类型
struct alignas(32) vec32_t { longlong4 val; };
struct alignas(64) vec64_t { longlong4 val[2]; };

template<int STEP_BYTES> struct VecTypeHelper;
template<> struct VecTypeHelper<8>  { using type = long long; };
template<> struct VecTypeHelper<16> { using type = longlong2; };
template<> struct VecTypeHelper<32> { using type = vec32_t; };
template<> struct VecTypeHelper<64> { using type = vec64_t; };

/**
 * @brief 模板化的 CUDA Kernel，用于测试任意粒度和跨度的读取带宽
 * @tparam STEP_BYTES 模板参数，指定每个线程单次读取的字节数 (one_step)
 */
template <int STEP_BYTES>
__global__ void stridedBandwidthKernel(const char* gmem_in, char* gmem_out, size_t num_reads, int stride_in_steps)
{
    // 使用辅助结构体获取与STEP_BYTES匹配的向量化类型
    using VecType = typename VecTypeHelper<STEP_BYTES>::type;

    // 将 char* 指针转换为向量化的类型指针
    const VecType* typed_in = reinterpret_cast<const VecType*>(gmem_in);
    VecType* typed_out = reinterpret_cast<VecType*>(gmem_out);

    // 计算当前线程的逻辑ID
    size_t logical_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (logical_idx < num_reads) {
        // 根据逻辑ID和跨度计算物理读取地址 (以VecType为单位)
        size_t read_idx = logical_idx * stride_in_steps;
        
        // 1. 从全局内存执行一次向量化的读取，数据直接进入寄存器
        VecType reg_val = typed_in[read_idx];

        // 2. 将寄存器中的数据写回，确保读取操作不被优化掉
        //    写入是连续的，以方便验证
        typed_out[logical_idx] = reg_val;
    }
}

// =================================================================================
// 2. 主机端 Main 函数
// =================================================================================
int main(int argc, char** argv)
{
    // --- 默认参数 ---
    int one_step_bytes = 8;
    int stride_bytes = 16;
    size_t total_gmem_size_bytes = 2ULL * 1024 * 1024 * 1024; // 2 GB
    int num_iterations = 100;

    // --- 从命令行解析参数 ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--step=", 0) == 0) {
            one_step_bytes = std::stoi(arg.substr(7));
        } else if (arg.rfind("--stride=", 0) == 0) {
            stride_bytes = std::stoi(arg.substr(9));
        } else if (arg.rfind("--iter=", 0) == 0) {
            num_iterations = std::stoi(arg.substr(7));
        }
    }

    // --- 参数验证 ---
    if (stride_bytes % one_step_bytes != 0) {
        std::cerr << "Error: Stride (" << stride_bytes << ") must be a multiple of step (" << one_step_bytes << ")." << std::endl;
        return -1;
    }

    // --- 计算启动参数 ---
    int stride_in_steps = stride_bytes / one_step_bytes;
    size_t num_total_reads = total_gmem_size_bytes / stride_bytes;
    int threads_per_block = 256;
    int blocks_per_grid = (num_total_reads + threads_per_block - 1) / threads_per_block;

    std::cout << "Starting Generic Strided Memory Bandwidth Benchmark..." << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Total GPU Memory Span  : " << total_gmem_size_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "Read Granularity (step): " << one_step_bytes << " Bytes" << std::endl;
    std::cout << "Stride                 : " << stride_bytes << " Bytes" << std::endl;
    std::cout << "Threads per Block      : " << threads_per_block << std::endl;
    std::cout << "Grid Size (Blocks)     : " << blocks_per_grid << std::endl;
    std::cout << "Total Reads per Trial  : " << num_total_reads << std::endl;
    std::cout << "Iterations             : " << num_iterations << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    
    // --- 内存分配 ---
    char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, total_gmem_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, num_total_reads * one_step_bytes)); 

    // --- 计时与 Kernel 启动 ---
    std::chrono::duration<double, std::milli> total_duration(0);
    
    // 使用 switch 语句根据运行时参数选择要启动的 Kernel 模板实例
    void (*kernel_ptr)(const char*, char*, size_t, int);

    switch (one_step_bytes) {
        case 8:  kernel_ptr = stridedBandwidthKernel<8>;  break;
        case 16: kernel_ptr = stridedBandwidthKernel<16>; break;
        case 32: kernel_ptr = stridedBandwidthKernel<32>; break;
        case 64: kernel_ptr = stridedBandwidthKernel<64>; break;
        default:
            std::cerr << "Error: Unsupported step size " << one_step_bytes << ". Supported sizes are 8, 16, 32, 64." << std::endl;
            return -1;
    }

    std::cout << "Running benchmark..." << std::endl;
    
    // 预热
    kernel_ptr<<<blocks_per_grid, threads_per_block>>>(d_in, d_out, num_total_reads, stride_in_steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < num_iterations; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        kernel_ptr<<<blocks_per_grid, threads_per_block>>>(d_in, d_out, num_total_reads, stride_in_steps);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end_time = std::chrono::high_resolution_clock::now();
        total_duration += (end_time - start_time);
    }
    
    // --- 计算并打印结果 ---
    double average_time_ms = total_duration.count() / num_iterations;
    double average_time_s = average_time_ms / 1000.0;
    double total_data_moved_bytes = 2.0 * num_total_reads * one_step_bytes;
    double effective_bandwidth_gbs = total_data_moved_bytes / (1024.0 * 1024.0 * 1024.0) / average_time_s;

    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average Kernel Time: " << average_time_ms << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << effective_bandwidth_gbs << " GB/s" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // --- 释放资源 ---
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}