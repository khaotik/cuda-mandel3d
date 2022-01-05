#pragma once
#include <cuda_runtime.h>
namespace curand {
#include <curand.h>
}

#define DEBUG_BUILD
#define CUDA_KERNEL __global__ void
#define CUDA_FN inline __device__ __host__

template<typename T> inline auto
cudaMallocTyped(T **ptr, size_t size=1) {
  return cudaMalloc(ptr, size*sizeof(T));
}
template<typename err_t> inline
err_t checkCudaError(const err_t &err);

template<> inline cudaError_t
checkCudaError<cudaError_t>(const cudaError_t &err) {
  if (err != cudaSuccess) {
    printf(
        "CUDA error[%s]: %s\n",
        cudaGetErrorName(err),
        cudaGetErrorString(err));
  }
  return err;
}
inline cudaError_t
checkCudaError() {
  return checkCudaError(cudaGetLastError());
}

template<> inline curand::curandStatus_t
checkCudaError<curand::curandStatus_t>(const curand::curandStatus_t &err) {
  if (err != curand::CURAND_STATUS_SUCCESS) {
    printf( "CURAND error[%d]\n", err);
  }
  return err;
}

template<typename inp_t>
struct scan_typinf {
  typedef inp_t ret_t;
};
template<>
struct scan_typinf<bool> {
  typedef unsigned ret_t;
};

template <typename inp_t> __device__ auto
warp_accum_sum(inp_t x, unsigned lane_idx) {
  inp_t sum = x;
  #pragma unroll
  for(int bit=1; bit!=32; bit<<=1) {
    inp_t x2 = __shfl_sync(0xffffffffU, sum, (lane_idx-bit)|(bit-1));
    if (lane_idx & bit) sum += x2;
  }
  return sum;
}

template<> __device__ auto
warp_accum_sum<bool>(bool x, unsigned lane_idx) {
  unsigned bits = __ballot_sync(0xffffffffU, x);
  bits &= (2U << lane_idx)-1;
  return __popc(bits);
}

template <typename inp_t>
__device__  auto block_accum_sum(
    inp_t x, unsigned lane_idx, unsigned warp_idx,
    typename scan_typinf<inp_t>::ret_t warp_sum_shm[32]) {
  using ret_t = typename scan_typinf<inp_t>::ret_t;
  ret_t warp_sum = warp_accum_sum(x, lane_idx);
  if (lane_idx == 31) warp_sum_shm[warp_idx] = warp_sum;
  __syncthreads();
  if (warp_idx == 0) {
    warp_sum_shm[lane_idx] = warp_accum_sum(warp_sum_shm[lane_idx], lane_idx);
  }
  __syncthreads();
  if (warp_idx != 0) {
    warp_sum += warp_sum_shm[warp_idx-1];
  }
  return warp_sum;
}
