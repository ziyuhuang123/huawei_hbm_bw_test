#include <iostream>
#include <iomanip>

// 用於 CUDA 錯誤檢查的宏
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) \
                  << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    std::cout << "Starting CUDA Environment Health Check..." << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "Error: No CUDA-capable devices were detected." << std::endl;
        return -1;
    }

    std::cout << "Found " << device_count << " CUDA device(s)." << std::endl;

    // 檢查第一個設備 (Device 0)
    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    std::cout << "Checking properties for Device " << device_id << "..." << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));

    std::cout << "Device Name                  : " << props.name << std::endl;
    std::cout << "Compute Capability           : " << props.major << "." << props.minor << std::endl;
    std::cout << "Total Global Memory          : " << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    
    // 再次嘗試查詢共享記憶體大小
    int max_smem_per_block;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));

    int max_smem_per_multiprocessor;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_multiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id));

    std::cout << "Max Shared Memory per Block  : " << max_smem_per_block << " Bytes" << std::endl;
    std::cout << "Max Shared Memory per SM     : " << max_smem_per_multiprocessor << " Bytes" << std::endl;
    
    std::cout << "------------------------------------------------------" << std::endl;

    if (max_smem_per_block == 0) {
        std::cout << "Health Check Result: FAILED. The driver returned 0 for max shared memory." << std::endl;
        std::cout << "This indicates a critical issue with the CUDA runtime/driver state." << std::endl;
    } else {
        std::cout << "Health Check Result: PASSED. The CUDA environment appears to be OK." << std::endl;
    }

    // 重置設備以清理 context
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}