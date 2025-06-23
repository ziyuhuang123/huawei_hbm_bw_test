// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <iomanip>
// #include <string>
// #include <stdexcept>
// #include <numeric>

// // 用于错误检查的宏
// #define CUDA_CHECK(err) { \
//     cudaError_t err_code = err; \
//     if (err_code != cudaSuccess) { \
//         std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) \
//                   << " at line " << __LINE__ << std::endl; \
//         exit(EXIT_FAILURE); \
//     } \
// }

// // 向量化类型定义 (保持不变)
// struct alignas(32) vec32_t { longlong4 val; };
// struct alignas(64) vec64_t { longlong4 val[2]; };

// template<int STEP_BYTES> struct VecTypeHelper;
// template<> struct VecTypeHelper<8>  { using type = long long; };
// template<> struct VecTypeHelper<16> { using type = longlong2; };
// template<> struct VecTypeHelper<32> { using type = vec32_t; };
// template<> struct VecTypeHelper<64> { using type = vec64_t; };

// /**
//  * @brief 模板化的 CUDA Kernel (GMEM -> REG -> SMEM)
//  * @tparam READS_PER_THREAD 每个线程的读取次数，作为编译期常量
//  */
// template <int STEP_BYTES, int READS_PER_THREAD>
// __global__ void stridedBandwidthKernel(const char* gmem_in, size_t num_total_reads, int stride_in_steps)
// {
//     // 声明动态共享内存
//     extern __shared__ char smem_buffer[];

//     using VecType = typename VecTypeHelper<STEP_BYTES>::type;

//     const VecType* typed_in = reinterpret_cast<const VecType*>(gmem_in);
//     VecType* typed_smem = reinterpret_cast<VecType*>(smem_buffer);

//     // =======================================================================
//     //          --- 核心修正: 实现 Grid-Stride Loop 模式 ---
//     // =======================================================================

//     // 1. 计算当前线程在整个 Grid 中的唯一、线性的 ID
//     const size_t global_thread_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    
//     // 2. 计算整个 Grid 的总线程数 (这就是我们的“大跨步”步长)
//     const size_t grid_size = static_cast<size_t>(gridDim.x) * blockDim.x;

//     // 3. 使用 #pragma unroll 指令展开循环
//     //    这个循环现在完全符合您的要求
//     #pragma unroll
//     for (int i = 0; i < READS_PER_THREAD; ++i) {
//         // 计算当前循环迭代的逻辑读取索引
//         // 第一次循环 (i=0), th0 读 0, th1 读 1 ...
//         // 第二次循环 (i=1), th0 读 grid_size, th1 读 grid_size+1 ...
//         const size_t current_logical_idx = global_thread_id + static_cast<size_t>(i) * grid_size;

//         if (current_logical_idx < num_total_reads) {
//             const size_t read_idx = current_logical_idx * stride_in_steps;
//             const VecType reg_val = typed_in[read_idx];
//             typed_smem[threadIdx.x] = reg_val;
//         }
//     }
    
//     __syncthreads();
// }

// // =======================================================================
// // 核心修改点: Kernel 启动分发器
// // 根据运行时的参数，选择正确的模板化 Kernel 实例
// // =======================================================================
// void launch_kernel(
//     int one_step_bytes, int reads_per_thread,
//     dim3 grid, dim3 block, size_t smem_size,
//     const char* d_in, size_t num_total_reads, int stride_in_steps) 
// {
//     switch (one_step_bytes) {
//         case 8:
//             switch (reads_per_thread) {
//                 case 1:  stridedBandwidthKernel<8, 1> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 2:  stridedBandwidthKernel<8, 2> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 4:  stridedBandwidthKernel<8, 4> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 8:  stridedBandwidthKernel<8, 8> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 16: stridedBandwidthKernel<8, 16><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 32: stridedBandwidthKernel<8, 32><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 64: stridedBandwidthKernel<8, 64><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 128:stridedBandwidthKernel<8, 128><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 256:stridedBandwidthKernel<8, 256><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 default: throw std::runtime_error("Unsupported reads_per_thread for step=8");
//             }
//             break;
//         case 16:
//              switch (reads_per_thread) {
//                 case 1:  stridedBandwidthKernel<16, 1> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 2:  stridedBandwidthKernel<16, 2> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 4:  stridedBandwidthKernel<16, 4> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 8:  stridedBandwidthKernel<16, 8> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 16: stridedBandwidthKernel<16, 16><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 32: stridedBandwidthKernel<16, 32><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 64: stridedBandwidthKernel<16, 64><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 128:stridedBandwidthKernel<16, 128><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 256:stridedBandwidthKernel<16, 256><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 default: throw std::runtime_error("Unsupported reads_per_thread for step=16");
//             }
//             break;
//         case 32:
//              switch (reads_per_thread) {
//                 case 1:  stridedBandwidthKernel<32, 1> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 2:  stridedBandwidthKernel<32, 2> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 4:  stridedBandwidthKernel<32, 4> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 8:  stridedBandwidthKernel<32, 8> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 16: stridedBandwidthKernel<32, 16><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 32: stridedBandwidthKernel<32, 32><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 64: stridedBandwidthKernel<32, 64><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 128:stridedBandwidthKernel<32, 128><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 256:stridedBandwidthKernel<32, 256><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 default: throw std::runtime_error("Unsupported reads_per_thread for step=32");
//             }
//             break;
//         case 64:
//              switch (reads_per_thread) {
//                 case 1:  stridedBandwidthKernel<64, 1> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 2:  stridedBandwidthKernel<64, 2> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 4:  stridedBandwidthKernel<64, 4> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 8:  stridedBandwidthKernel<64, 8> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 16: stridedBandwidthKernel<64, 16><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 32: stridedBandwidthKernel<64, 32><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 64: stridedBandwidthKernel<64, 64><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 128:stridedBandwidthKernel<64, 128><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 case 256:stridedBandwidthKernel<64, 256><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
//                 default: throw std::runtime_error("Unsupported reads_per_thread for step=64");
//             }
//             break;
//         default:
//             throw std::runtime_error("Unsupported step size");
//     }
// }


// // 主机端 Main 函数
// int main(int argc, char** argv)
// {
//     int one_step_bytes = 8;
//     int stride_bytes = 8;
//     // size_t total_gmem_size_bytes = 2ULL * 1024 * 1024 * 1024;
//     size_t total_gmem_size_bytes = 64ULL * 1024 * 1024 * 1024; // 修改後 (32GB)
//     int num_iterations = 100;
//     int reads_per_thread = 8; 

//     for (int i = 1; i < argc; ++i) {
//         std::string arg = argv[i];
//         if (arg.rfind("--step=", 0) == 0) { one_step_bytes = std::stoi(arg.substr(7)); } 
//         else if (arg.rfind("--stride=", 0) == 0) { stride_bytes = std::stoi(arg.substr(9)); } 
//         else if (arg.rfind("--iter=", 0) == 0) { num_iterations = std::stoi(arg.substr(7)); } 
//         else if (arg.rfind("--reads_per_thread=", 0) == 0) { reads_per_thread = std::stoi(arg.substr(19)); }
//     }

//     if (stride_bytes % one_step_bytes != 0) {
//         std::cerr << "Error: Stride (" << stride_bytes << ") must be a multiple of step (" << one_step_bytes << ")." << std::endl;
//         return -1;
//     }

//     int stride_in_steps = stride_bytes / one_step_bytes;
//     size_t num_total_reads = total_gmem_size_bytes / stride_bytes;
//     int threads_per_block = 512;

//     size_t total_threads_to_launch = (num_total_reads + reads_per_thread - 1) / reads_per_thread;
//     int blocks_per_grid = (total_threads_to_launch + threads_per_block - 1) / threads_per_block;

//     std::cout << "Starting Templated/Unrolled Strided Memory Bandwidth Benchmark..." << std::endl;
//     std::cout << "------------------------------------------------------" << std::endl;
//     std::cout << "Total GPU Memory Span  : " << total_gmem_size_bytes / (1024*1024) << " MB" << std::endl;
//     std::cout << "Read Granularity (step): " << one_step_bytes << " Bytes" << std::endl;
//     std::cout << "Stride                 : " << stride_bytes << " Bytes" << std::endl;
//     std::cout << "Reads per Thread (TPL) : " << reads_per_thread << " (Supported: 1,2,4,8,16,32)" << std::endl;
//     std::cout << "Threads per Block      : " << threads_per_block << std::endl;
//     std::cout << "Grid Size (Blocks)     : " << blocks_per_grid << std::endl;
//     std::cout << "Total Reads per Trial  : " << num_total_reads << std::endl;
//     std::cout << "Iterations             : " << num_iterations << std::endl;
    
//     char *d_in;
//     CUDA_CHECK(cudaMalloc(&d_in, total_gmem_size_bytes));
//     CUDA_CHECK(cudaMemset(d_in, 1, total_gmem_size_bytes));

//     size_t smem_size_bytes = threads_per_block * one_step_bytes;
    
//     int max_smem_per_block;
//     CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0));
//     if (smem_size_bytes > max_smem_per_block) {
//         std::cerr << "Error: SMEM request (" << smem_size_bytes << ") > device limit (" << max_smem_per_block << ")." << std::endl;
//         return -1;
//     }
//     std::cout << "Shared Memory per Block: " << smem_size_bytes << " Bytes" << std::endl;
//     std::cout << "------------------------------------------------------" << std::endl;

//     std::chrono::duration<double, std::milli> total_duration(0);

//     dim3 grid(blocks_per_grid, 1, 1);
//     dim3 block(threads_per_block, 1, 1);

//     try {
//         std::cout << "Running benchmark..." << std::endl;
        
//         // 预热/JIT编译
//         launch_kernel(one_step_bytes, reads_per_thread, grid, block, smem_size_bytes, d_in, num_total_reads, stride_in_steps);
//         CUDA_CHECK(cudaGetLastError());
//         CUDA_CHECK(cudaDeviceSynchronize());

//         for (int i = 0; i < num_iterations; ++i) {
//             auto start_time = std::chrono::high_resolution_clock::now();
//             launch_kernel(one_step_bytes, reads_per_thread, grid, block, smem_size_bytes, d_in, num_total_reads, stride_in_steps);
//             CUDA_CHECK(cudaDeviceSynchronize());
//             auto end_time = std::chrono::high_resolution_clock::now();
//             total_duration += (end_time - start_time);
//         }
//     } catch (const std::runtime_error& e) {
//         std::cerr << "Execution Error: " << e.what() << std::endl;
//         std::cerr << "Please choose a supported value for --reads_per_thread (e.g., 1, 2, 4, 8, 16, 32)." << std::endl;
//         cudaFree(d_in);
//         cudaDeviceReset();
//         return -1;
//     }
    
//     double average_time_ms = total_duration.count() / num_iterations;
//     double average_time_s = average_time_ms / 1000.0;
    
//     double total_data_read_bytes = 1.0 * num_total_reads * one_step_bytes;
//     double read_bandwidth_gbs = total_data_read_bytes / (1024.0 * 1024.0 * 1024.0) / average_time_s;

//     std::cout << "\nBenchmark Results:" << std::endl;
//     std::cout << "------------------------------------------------------" << std::endl;
//     std::cout << std::fixed << std::setprecision(3);
//     std::cout << "Average Kernel Time: " << average_time_ms << " ms" << std::endl;
//     std::cout << "Read-Only Bandwidth: " << read_bandwidth_gbs << " GB/s" << std::endl;
//     std::cout << "------------------------------------------------------" << std::endl;

//     CUDA_CHECK(cudaFree(d_in));
//     CUDA_CHECK(cudaDeviceReset());

//     return 0;
// } // nvcc -o test-new test-new.cu -O3 -std=c++17 -arch=sm_90a




#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <numeric>

// 用於錯誤檢查的宏
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) \
                  << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// 向量化類型定義
struct alignas(32)  vec32_t  { longlong4 val; };
struct alignas(64)  vec64_t  { longlong4 val[2]; };
struct alignas(128) vec128_t { longlong4 val[4]; };

// 類型輔助模板
template<int STEP_BYTES> struct VecTypeHelper;
template<> struct VecTypeHelper<8>   { using type = long long; };
template<> struct VecTypeHelper<16>  { using type = longlong2; };
template<> struct VecTypeHelper<32>  { using type = vec32_t; };
template<> struct VecTypeHelper<64>  { using type = vec64_t; };
template<> struct VecTypeHelper<128> { using type = vec128_t; };

/**
 * @brief 模板化的 CUDA Kernel (GMEM -> REG -> SMEM)
 */
template <int STEP_BYTES, int READS_PER_THREAD>
__global__ void stridedBandwidthKernel(const char* gmem_in, size_t num_total_reads, int stride_in_steps)
{
    extern __shared__ char smem_buffer[];
    using VecType = typename VecTypeHelper<STEP_BYTES>::type;

    const VecType* typed_in = reinterpret_cast<const VecType*>(gmem_in);
    VecType* typed_smem = reinterpret_cast<VecType*>(smem_buffer);

    const size_t global_thread_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t grid_size = static_cast<size_t>(gridDim.x) * blockDim.x;

    #pragma unroll
    for (int i = 0; i < READS_PER_THREAD; ++i) {
        const size_t current_logical_idx = global_thread_id + static_cast<size_t>(i) * grid_size;
        if (current_logical_idx < num_total_reads) {
            const size_t read_idx = current_logical_idx * stride_in_steps;
            const VecType reg_val = typed_in[read_idx];
            typed_smem[threadIdx.x] = reg_val;
        }
    }
    __syncthreads();
}

/**
 * @brief Kernel 啟動分發器
 */
void launch_kernel(
    int one_step_bytes, int reads_per_thread,
    dim3 grid, dim3 block, size_t smem_size,
    const char* d_in, size_t num_total_reads, int stride_in_steps) 
{
    switch (one_step_bytes) {
        case 8:
            switch (reads_per_thread) {
                case 1:   stridedBandwidthKernel<8, 1> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 2:   stridedBandwidthKernel<8, 2> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 4:   stridedBandwidthKernel<8, 4> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 8:   stridedBandwidthKernel<8, 8> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 16:  stridedBandwidthKernel<8, 16><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 32:  stridedBandwidthKernel<8, 32><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 64:  stridedBandwidthKernel<8, 64><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 128: stridedBandwidthKernel<8, 128><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 256: stridedBandwidthKernel<8, 256><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                default: throw std::runtime_error("Unsupported reads_per_thread for step=8");
            }
            break;
        case 16:
             switch (reads_per_thread) {
                case 1:   stridedBandwidthKernel<16, 1> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 2:   stridedBandwidthKernel<16, 2> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 4:   stridedBandwidthKernel<16, 4> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 8:   stridedBandwidthKernel<16, 8> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 16:  stridedBandwidthKernel<16, 16><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 32:  stridedBandwidthKernel<16, 32><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 64:  stridedBandwidthKernel<16, 64><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 128: stridedBandwidthKernel<16, 128><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 256: stridedBandwidthKernel<16, 256><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                default: throw std::runtime_error("Unsupported reads_per_thread for step=16");
            }
            break;
        case 32:
             switch (reads_per_thread) {
                case 1:   stridedBandwidthKernel<32, 1> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 2:   stridedBandwidthKernel<32, 2> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 4:   stridedBandwidthKernel<32, 4> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 8:   stridedBandwidthKernel<32, 8> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 16:  stridedBandwidthKernel<32, 16><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 32:  stridedBandwidthKernel<32, 32><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 64:  stridedBandwidthKernel<32, 64><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 128: stridedBandwidthKernel<32, 128><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 256: stridedBandwidthKernel<32, 256><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                default: throw std::runtime_error("Unsupported reads_per_thread for step=32");
            }
            break;
        case 64:
             switch (reads_per_thread) {
                case 1:   stridedBandwidthKernel<64, 1> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 2:   stridedBandwidthKernel<64, 2> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 4:   stridedBandwidthKernel<64, 4> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 8:   stridedBandwidthKernel<64, 8> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 16:  stridedBandwidthKernel<64, 16><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 32:  stridedBandwidthKernel<64, 32><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 64:  stridedBandwidthKernel<64, 64><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 128: stridedBandwidthKernel<64, 128><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 256: stridedBandwidthKernel<64, 256><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                default: throw std::runtime_error("Unsupported reads_per_thread for step=64");
            }
            break;
        case 128:
             switch (reads_per_thread) {
                case 1:   stridedBandwidthKernel<128, 1> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 2:   stridedBandwidthKernel<128, 2> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 4:   stridedBandwidthKernel<128, 4> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 8:   stridedBandwidthKernel<128, 8> <<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 16:  stridedBandwidthKernel<128, 16><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 32:  stridedBandwidthKernel<128, 32><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 64:  stridedBandwidthKernel<128, 64><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 128: stridedBandwidthKernel<128, 128><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                case 256: stridedBandwidthKernel<128, 256><<<grid, block, smem_size>>>(d_in, num_total_reads, stride_in_steps); break;
                default: throw std::runtime_error("Unsupported reads_per_thread for step=128");
            }
            break;
        default:
            throw std::runtime_error("Unsupported step size. Supported: 8, 16, 32, 64, 128");
    }
}

// 主機端 Main 函數
int main(int argc, char** argv)
{
    // 1. 參數與變數初始化
    int one_step_bytes = 128;
    int stride_bytes = 128;
    size_t total_gmem_size_bytes = 32ULL * 1024 * 1024 * 1024; // 32 GB
    int num_iterations = 100;
    int reads_per_thread = 8;
    int threads_per_block = 1024; // 預設一個理想值

    // 2. 命令列參數解析
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--step=", 0) == 0) { one_step_bytes = std::stoi(arg.substr(7)); } 
        else if (arg.rfind("--stride=", 0) == 0) { stride_bytes = std::stoi(arg.substr(9)); } 
        else if (arg.rfind("--iter=", 0) == 0) { num_iterations = std::stoi(arg.substr(7)); } 
        else if (arg.rfind("--reads_per_thread=", 0) == 0) { reads_per_thread = std::stoi(arg.substr(19)); }
        else if (arg.rfind("--tpb=", 0) == 0) { threads_per_block = std::stoi(arg.substr(6)); }
    }

    if (stride_bytes % one_step_bytes != 0) {
        std::cerr << "Error: Stride (" << stride_bytes << ") must be a multiple of step (" << one_step_bytes << ")." << std::endl;
        return -1;
    }

    // 3. 針對 H100 等高階GPU，請求最大的共享內存 (Opt-In)
    CUDA_CHECK(cudaSetDevice(0));
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    
    int max_optin_smem_per_block;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Device Opt-In Shared Memory per Block: " << max_optin_smem_per_block << " Bytes." << std::endl;

    if (max_optin_smem_per_block > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            stridedBandwidthKernel<128, 8>, // 選擇一個具體實例來設置屬性
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            max_optin_smem_per_block
        ));
        std::cout << "Kernel attribute set to allow up to " << max_optin_smem_per_block << " B of dynamic shared memory." << std::endl;
    }

    // 4. 分配設備內存
    char *d_in;
    CUDA_CHECK(cudaMalloc(&d_in, total_gmem_size_bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, total_gmem_size_bytes));

    // 5. 根據最終的共享內存上限，自動調整執行緒數量
    int current_max_smem;
    CUDA_CHECK(cudaDeviceGetAttribute(&current_max_smem, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    
    size_t required_smem_per_block = static_cast<size_t>(threads_per_block) * one_step_bytes;
    bool adjusted = false;

    if (required_smem_per_block > current_max_smem) {
        int old_tpb = threads_per_block;
        threads_per_block = current_max_smem / one_step_bytes;
        threads_per_block = (threads_per_block / 32) * 32; // 向下取整到Warp大小的倍數
        adjusted = true;

        if (threads_per_block == 0) {
            std::cerr << "Error: Step size (" << one_step_bytes << "B) is too large for even a single warp to fit in the SMEM limit (" << current_max_smem << "B)." << std::endl;
            cudaFree(d_in);
            return -1;
        }
        std::cout << "ADJUSTMENT: SMEM request > limit. Threads per block adjusted from " << old_tpb << " -> " << threads_per_block << std::endl;
    }

    // 6. 準備並執行 Benchmark
    int stride_in_steps = stride_bytes / one_step_bytes;
    size_t num_total_reads = total_gmem_size_bytes / stride_bytes;
    size_t total_threads_to_launch = (num_total_reads + reads_per_thread - 1) / reads_per_thread;
    int blocks_per_grid = (total_threads_to_launch + threads_per_block - 1) / threads_per_block;
    size_t smem_size_bytes = static_cast<size_t>(threads_per_block) * one_step_bytes;

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Benchmark Configuration:" << std::endl;
    std::cout << "  Read Granularity (step): " << one_step_bytes << " Bytes" << std::endl;
    std::cout << "  Threads per Block      : " << threads_per_block << (adjusted ? " (Adjusted)" : "") << std::endl;
    std::cout << "  Shared Memory per Block: " << smem_size_bytes << " Bytes (Limit: " << current_max_smem << " B)" << std::endl;
    std::cout << "  Grid Size (Blocks)     : " << blocks_per_grid << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    dim3 grid(blocks_per_grid, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    std::chrono::duration<double, std::milli> total_duration(0);

    try {
        std::cout << "Running benchmark..." << std::endl;
        launch_kernel(one_step_bytes, reads_per_thread, grid, block, smem_size_bytes, d_in, num_total_reads, stride_in_steps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < num_iterations; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();
            launch_kernel(one_step_bytes, reads_per_thread, grid, block, smem_size_bytes, d_in, num_total_reads, stride_in_steps);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end_time = std::chrono::high_resolution_clock::now();
            total_duration += (end_time - start_time);
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Execution Error: " << e.what() << std::endl;
        cudaFree(d_in);
        cudaDeviceReset();
        return -1;
    }

    // 7. 顯示結果並清理
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

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}