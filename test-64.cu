#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>

// 用於 CUDA 錯誤檢查的宏
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) \
                  << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ======================================================
//  核心修改 (1): 將讀取類型從 128B 改為 64B
// ======================================================
struct alignas(64) ReadType64B {
    longlong4 data[2]; // longlong4 是 32 bytes, 2 * 32 = 64 bytes
};

__global__ void raceConditionAddKernel64B(const ReadType64B* d_in, unsigned long long total_reads)
{
    extern __shared__ unsigned long long smem_accumulator[];

    if (threadIdx.x == 0) {
        smem_accumulator[0] = 0;
    }
    __syncthreads();

    const unsigned long long global_thread_id = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_id >= total_reads) { return; }

    // 1. 從 Global Memory 讀取 64B 數據
    ReadType64B reg_val = d_in[global_thread_id];

    // 2. 在 Register 中計算局部和
    unsigned long long local_sum = 0;
    // ======================================================
    //  核心修改 (2): 更新加總邏輯以匹配 64B 結構
    // ======================================================
    local_sum += reg_val.data[0].x + reg_val.data[0].y + reg_val.data[0].z + reg_val.data[0].w;
    local_sum += reg_val.data[1].x + reg_val.data[1].y + reg_val.data[1].z + reg_val.data[1].w;

    // 警告：仍然是帶有競爭條件的非原子性累加
    smem_accumulator[0] += local_sum;
}

int main(int argc, char** argv)
{
    // ======================================================
    //  核心修改 (3): 將每個線程的位元組數改為 64
    // ======================================================
    const int BYTES_PER_THREAD = 64;
    int threads_per_block = 256;
    long long total_gmem_size_bytes = 4LL * 1024 * 1024 * 1024; // 4 GB
    int num_iterations = 100;
    
    std::cout << "Starting Race Condition (+= to Same Location) 64B Benchmark..." << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Read Granularity   : " << BYTES_PER_THREAD << " Bytes/Thread" << std::endl;
    std::cout << "Threads per Block  : " << threads_per_block << std::endl;
    std::cout << "Total Memory Size  : " << total_gmem_size_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // 分配輸入緩衝區
    ReadType64B* d_in; // <-- 類型更新
    CUDA_CHECK(cudaMalloc(&d_in, total_gmem_size_bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, total_gmem_size_bytes));

    // 計算 Kernel 啟動配置
    unsigned long long num_total_reads = total_gmem_size_bytes / BYTES_PER_THREAD;
    int blocks_per_grid = (num_total_reads + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_per_grid, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    std::cout << "Grid size: " << grid.x << " blocks, Block size: " << block.x << " threads" << std::endl;

    // 共享記憶體大小不變，仍然是一個 unsigned long long
    size_t smem_size_bytes = sizeof(unsigned long long);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 預熱
    raceConditionAddKernel64B<<<grid, block, smem_size_bytes>>>(d_in, num_total_reads);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 計時
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        raceConditionAddKernel64B<<<grid, block, smem_size_bytes>>>(d_in, num_total_reads);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 計算並顯示結果
    float elapsed_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, stop));
    
    double total_data_processed_gb = (double)total_gmem_size_bytes * num_iterations / (1024.0 * 1024.0 * 1024.0);
    double total_time_s = elapsed_time_ms / 1000.0;
    double read_bandwidth_gbs = total_data_processed_gb / total_time_s;
    double average_time_s = total_time_s / num_iterations;

    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average Kernel Time  : " << average_time_s * 1000.0 << " ms" << std::endl;
    std::cout << "Achieved Read BW     : " << read_bandwidth_gbs << " GB/s" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));

    return 0;
}