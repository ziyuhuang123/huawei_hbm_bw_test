// nvcc -o tma_stride_demo tma_stride_demo.cu --std=c++17 -O3 -arch=sm_90 -lcuda && ./tma_stride_demo  华为给我的原版文件

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

int tma_test(int sm_last_dim, int gm_last_dim) {
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
  return 0;
}


int main(int argc, char** argv) {
  tma_test(8, 32);
  tma_test(16, 32);
  tma_test(32, 32);
  tma_test(32, 64);
}