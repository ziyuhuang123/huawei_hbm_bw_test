#include <iostream>

#include "util.h"


constexpr int n_buffer = 1;

__global__ void test_tma_2(uint32_t* input, uint32_t* output, CUtensorMap* input_desc,
                           int64_t gm_first_dim, int64_t gm_last_dim,
                           int64_t sm_fist_dim, int64_t sm_last_dim) {

  extern __shared__ char sm_base[];
  uint32_t* sm_buffer = (uint32_t*)sm_base;
  uint64_t* sm_mbarrier = (uint64_t*)(192*1024);
  uint64_t* sm_reverse_mbarrier = (uint64_t*)(192*1024 + 1024);

  // one thread per warp
  if (threadIdx.y == 0 && threadIdx.x % 32 != 0) {
    return;
  }

  uint32_t dummy_sum = 0.0;

  if (threadIdx.y == 0) {
    for (int i = 0; i <n_buffer; ++i) {
      initialize_barrier(sm_mbarrier[i]);
      initialize_barrier(sm_reverse_mbarrier[i], 32);
    }
  }
  __syncthreads();

  int phase[n_buffer];
  for (int i = 0; i < n_buffer; ++i) {
    phase[i] = 0;
  }

  int buffer_id = 0;

  for (int64_t i = blockIdx.x * sm_fist_dim; i < gm_first_dim; i += gridDim.x * sm_fist_dim) {
    if (threadIdx.y == 0) {
      set_barrier_transaction_bytes(sm_mbarrier[buffer_id], sm_fist_dim * sm_last_dim * sizeof(uint32_t));
      SM90_TMA_LOAD_2D::copy(input_desc, sm_mbarrier[buffer_id],
		    &sm_buffer[buffer_id * sm_fist_dim * sm_last_dim],
        0, i);
      wait_barrier(sm_reverse_mbarrier[buffer_id], phase[buffer_id]);
    }
    else {
      wait_barrier(sm_mbarrier[buffer_id], phase[buffer_id]);
      arrive_barrier(sm_reverse_mbarrier[buffer_id]);
    }
    phase[buffer_id] = (phase[buffer_id] + 1)%2;
    buffer_id = (buffer_id + 1) % n_buffer;
  }
}

double tma_test(int sm_last_dim, int gm_last_dim) {
  int64_t last_dim = gm_last_dim;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  uint64_t sm_count = prop.multiProcessorCount;

  int64_t buffer_size = 1024ULL*1024*1024;
  int64_t first_dim = buffer_size / last_dim;
  int64_t sm_data_size = 192*1024;
  int64_t sm_fist_dim = sm_data_size / sizeof(uint32_t) / sm_last_dim / n_buffer;
  if (sm_fist_dim > 256) {
    sm_fist_dim = 256;
  }

  uint32_t* data_input_device = nullptr;
  cudaMalloc(&data_input_device, buffer_size*sizeof(uint32_t));
  uint32_t* data_output_device = nullptr;
  cudaMalloc(&data_output_device, buffer_size*sizeof(uint32_t));
  

  CUtensorMap input_tma_desc;

  CUtensorMap* input_tma_desc_device;
  cudaMalloc(&input_tma_desc_device, sizeof(CUtensorMap));

  uint64_t input_globalDim[5] = {1,1,1,1,1};
  uint64_t input_globalStride[5] = {0,0,0,0,0};

  input_globalDim[0] = last_dim;
  input_globalDim[1] = first_dim;

  input_globalStride[1] = last_dim * sizeof(uint32_t);

  uint32_t smem_box_shape[5] = {1,1,1,1,1};
  uint32_t smem_box_stride[5] = {1,1,1,1,1};

  smem_box_shape[0] = sm_last_dim;
  smem_box_shape[1] = sm_fist_dim;

  CUresult encode_result =
  cuTensorMapEncodeTiled(&input_tma_desc, 
		         CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
			 2,
			 data_input_device,
			 input_globalDim,
			 input_globalStride + 1,
			 smem_box_shape,
			 smem_box_stride,
			 CU_TENSOR_MAP_INTERLEAVE_NONE,
			 CU_TENSOR_MAP_SWIZZLE_NONE,
			 CU_TENSOR_MAP_L2_PROMOTION_NONE,
			 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (encode_result != CUDA_SUCCESS) {
	  std::cerr << "failed to init TMA desc\n";
	  return -1;
  }

  cudaMemcpy(input_tma_desc_device, &input_tma_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  int grid_dim = sm_count;
  dim3 block_dim(32, 2);

  size_t sm_size = 227*1024;

  cudaFuncSetAttribute(test_tma_2,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sm_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  test_tma_2<<<grid_dim, block_dim, sm_size>>>(data_input_device, data_output_device, input_tma_desc_device,
                                               first_dim, last_dim, sm_fist_dim, sm_last_dim);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  gpuErrchk(cudaDeviceSynchronize());



  double rd_bytes = first_dim * sm_last_dim * sizeof(uint32_t);
  double rd_gb = rd_bytes / 1024 / 1024 / 1024;
  double sec = time / 1000;

  double bw = rd_gb / sec;
  printf("read gm burst_len %d byte stride %d byte, bandwidth = %.1fGB/s\n", 
    (int)(sm_last_dim*sizeof(uint32_t)), (int)(last_dim*sizeof(uint32_t)), bw);

  cudaFree(data_input_device);
  cudaFree(data_output_device);
  cudaFree(input_tma_desc_device);
  return bw;
}


int main(int argc, char** argv) {
    // 1. 定义实验参数
    const int step_bytes = 64; // step 固定为 64B
    const int max_stride_bytes = 65536;
    const char* output_filename = "results.csv"; // 文件名统一为 results.csv

    // 2. 准备CSV文件并写入表头 "step,stride,bandwidth"
    FILE* fp = fopen(output_filename, "w");
    if (fp == NULL) {
        std::cerr << "错误: 无法打开输出文件: " << output_filename << std::endl;
        return 1;
    }
    fprintf(fp, "step,stride,bandwidth\n");

    // 3. 循环执行实验，stride倍数按2的幂次增加
    std::cout << "实验开始，结果将被写入 " << output_filename << std::endl;
    for (int multiple = 1; ; multiple *= 2) {
        int stride_bytes = multiple * step_bytes;
        if (stride_bytes > max_stride_bytes) {
            break; 
        }

        std::cout << "正在测试: step=" << step_bytes << " B, stride=" << stride_bytes << " B..." << std::endl;

        // 4. 计算参数并调用 tma_test
        int sm_last_dim = step_bytes / 4; 
        int gm_last_dim = stride_bytes / 4;
        

// tma_test 函数需要的参数 sm_last_dim 和 gm_last_dim，以及TMA描述符在定义内存布局时，它们描述的都是“有多少个数据元素”，而不是“有多少字节”。
// 这个 / 4 的操作，是在进行一个单位换算：将以 “字节”（Bytes） 为单位的长度，转换为以 “元素个数” 为单位的长度。


        double bandwidth = tma_test(sm_last_dim, gm_last_dim);

        // 5. 将结果写入文件
        fprintf(fp, "%d,%d,%.2f\n", step_bytes, stride_bytes, bandwidth);
    }

    fclose(fp);
    std::cout << "\n🎉 实验全部完成！" << std::endl;

    return 0;
}
// 以及这里生成的结果是和plot.py对应的results.csv。然后直接python plot.py就可以了。