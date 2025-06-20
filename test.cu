#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <numeric>

// 用于错误检查的宏
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 向量化类型定义 (保持不变)
struct alignas(32) vec32_t { longlong4 val; };
struct alignas(64) vec64_t { longlong4 val[2]; };

template<int STEP_BYTES> struct VecTypeHelper;
template<> struct VecTypeHelper<8>  { using type = long long; };
template<> struct VecTypeHelper<16> { using type = longlong2; };
template<> struct VecTypeHelper<32> { using type = vec32_t; };
template<> struct VecTypeHelper<64> { using type = vec64_t; };

/**
 * @brief 模板化的 CUDA Kernel (GMEM -> REG -> SMEM)
 */
template <int STEP_BYTES>
__global__ void stridedBandwidthKernel(const char* gmem_in, size_t num_reads, int stride_in_steps)
{
    // 声明动态共享内存
    extern __shared__ char smem_buffer[];

    using VecType = typename VecTypeHelper<STEP_BYTES>::type;

    // 将 char* 指针转换为向量化的类型指针
    const VecType* typed_in = reinterpret_cast<const VecType*>(gmem_in);
    VecType* typed_smem = reinterpret_cast<VecType*>(smem_buffer);

    // 计算当前线程的逻辑ID
    size_t logical_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (logical_idx < num_reads) {
        // 根据逻辑ID和跨度计算物理读取地址
        size_t read_idx = logical_idx * stride_in_steps;
        
        // 1. 从全局内存执行一次向量化的读取，数据直接进入寄存器
        VecType reg_val = typed_in[read_idx];

        // 2. 将数据从寄存器写入共享内存
        // 每个线程只在自己的 block 内写入，使用 threadIdx.x 作为索引
        typed_smem[threadIdx.x] = reg_val;
    }
    
    // 3. 线程块内同步
    // 这个屏障确保了写 SMEM 的操作是必须执行的，从而防止读 GMEM 被优化
    __syncthreads();
}

// 主机端 Main 函数
int main(int argc, char** argv)
{
    // --- 参数解析 (保持不变) ---
    int one_step_bytes = 8;
    int stride_bytes = 8;
    size_t total_gmem_size_bytes = 2ULL * 1024 * 1024 * 1024;
    int num_iterations = 100;

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

    if (stride_bytes % one_step_bytes != 0) {
        std::cerr << "Error: Stride (" << stride_bytes << ") must be a multiple of step (" << one_step_bytes << ")." << std::endl;
        return -1;
    }

    int stride_in_steps = stride_bytes / one_step_bytes;
    size_t num_total_reads = total_gmem_size_bytes / stride_bytes;
    int threads_per_block = 256;
    int blocks_per_grid = (num_total_reads + threads_per_block - 1) / threads_per_block;

    // =======================================================================
    // 核心修改点: 恢复打印配置信息
    // =======================================================================
    std::cout << "Starting Generic Strided Memory Bandwidth Benchmark (GMEM->SMEM)..." << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Total GPU Memory Span  : " << total_gmem_size_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "Read Granularity (step): " << one_step_bytes << " Bytes" << std::endl;
    std::cout << "Stride                 : " << stride_bytes << " Bytes" << std::endl;
    std::cout << "Threads per Block      : " << threads_per_block << std::endl;
    std::cout << "Grid Size (Blocks)     : " << blocks_per_grid << std::endl;
    std::cout << "Total Reads per Trial  : " << num_total_reads << std::endl;
    std::cout << "Iterations             : " << num_iterations << std::endl;
    
    // --- 内存分配、Kernel 签名和启动配置 ---
    char *d_in;
    CUDA_CHECK(cudaMalloc(&d_in, total_gmem_size_bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, total_gmem_size_bytes));

    size_t smem_size_bytes = threads_per_block * one_step_bytes;
    
    int max_smem_per_block;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    if (smem_size_bytes > max_smem_per_block) {
        std::cerr << "Error: Requested shared memory size (" << smem_size_bytes 
                  << " bytes) exceeds device limit (" << max_smem_per_block << " bytes)." << std::endl;
        return -1;
    }
    std::cout << "Shared Memory per Block: " << smem_size_bytes << " Bytes" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // --- 计时与 Kernel 启动 ---
    std::chrono::duration<double, std::milli> total_duration(0);
    
    void (*kernel_ptr)(const char*, size_t, int);

    switch (one_step_bytes) {
        case 8:  kernel_ptr = stridedBandwidthKernel<8>;  break;
        case 16: kernel_ptr = stridedBandwidthKernel<16>; break;
        case 32: kernel_ptr = stridedBandwidthKernel<32>; break;
        case 64: kernel_ptr = stridedBandwidthKernel<64>; break;
        default:
            std::cerr << "Error: Unsupported step size " << one_step_bytes << "." << std::endl;
            return -1;
    }

    std::cout << "Running benchmark..." << std::endl;
    
    kernel_ptr<<<blocks_per_grid, threads_per_block, smem_size_bytes>>>(d_in, num_total_reads, stride_in_steps);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < num_iterations; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        kernel_ptr<<<blocks_per_grid, threads_per_block, smem_size_bytes>>>(d_in, num_total_reads, stride_in_steps);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end_time = std::chrono::high_resolution_clock::now();
        total_duration += (end_time - start_time);
    }
    
    // --- 计算并打印结果 ---
    double average_time_ms = total_duration.count() / num_iterations;
    double average_time_s = average_time_ms / 1000.0;
    
    double total_data_read_bytes = 1.0 * num_total_reads * one_step_bytes;
    double read_bandwidth_gbs = total_data_read_bytes / (1024.0 * 1024.0 * 1024.0) / average_time_s;

    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average Kernel Time: " << average_time_ms << " ms" << std::endl;
    std::cout << "Read-Only Bandwidth: " << read_bandwidth_gbs << " GB/s" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // --- 释放资源 ---
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
