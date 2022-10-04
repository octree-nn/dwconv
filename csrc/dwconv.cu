#include "dwconv.h"
#include "utils.h"
#include <ATen/cuda/CUDAContext.h>

template <typename Dtype>
__device__ void block_reduce(Dtype* data) {
  int tid = threadIdx.x;
#pragma unroll
  for (int i = kCudaThreadsNum / 2; i > 0; i /= 2) {
    data[tid] += data[tid + i];
    __syncthreads();
  }
}

template <typename Dtype>
__global__ void dwconv_forward_backward_kernel(
    Dtype* out, const Dtype* __restrict__ data,
    const Dtype* __restrict__ weight, const int64_t* __restrict__ neigh,
    const int64_t height, const int64_t channel, const int64_t kngh,
    const int64_t nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int64_t h = i / channel;
    int64_t c = i % channel;
    Dtype value = Dtype(0);
    for (int k = 0; k < kngh; ++k) {
      int64_t ni = neigh[h * kngh + k];
      if (ni >= 0) {
        value += weight[k * channel + c] * data[ni * channel + c];
      }
    }
    out[i] = value;
  }
}

template <typename Dtype>
__global__ void dwconv_weight_backward_kernel(
    Dtype* out, const Dtype* __restrict__ grad, const Dtype* __restrict__ data,
    const int64_t* __restrict__ ineigh, const int64_t height, const int64_t channel,
    const int64_t kngh, const int64_t nthreads) {
  __shared__ Dtype weights[kCudaThreadsNum];
  CUDA_KERNEL_LOOP(i, nthreads) {
    int64_t h = i / channel;
    int64_t c = i % channel;
    for (int k = 0; k < kngh; ++k) {
      int64_t ni = ineigh[h * kngh + k];
      int64_t tid = threadIdx.x;
      weights[tid] = ni >= 0 ? data[ni * channel + c] * grad[i] : Dtype(0);
      __syncthreads();
      block_reduce(weights);
      if (tid == 0) {
        int n = i / kCudaThreadsNum;
        out[(n * kngh + k) * channel + c] = weights[0];
      }
    }
  }
}

__global__ void inverse_neigh_kernel(
    int64_t* ineigh, const int64_t* __restrict__ neigh, const int64_t height, 
    const int64_t kngh, const int64_t nthreads) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    int64_t h = i / kngh;
    int64_t c = i % kngh;
    int64_t j = neigh[i];
    if (j >= 0) {
      ineigh[j * kngh + c] = h;
    }
  }
}

Tensor inverse_neigh(Tensor neigh) {
  int64_t height = neigh.size(0);
  int64_t kngh = neigh.size(1);
  int64_t nthreads = height * kngh;
  Tensor ineigh = torch::full_like(neigh, -1);
  auto stream = at::cuda::getCurrentCUDAStream();
  inverse_neigh_kernel
      <<<CudaGetBlocks(nthreads), kCudaThreadsNum, 0, stream>>>(
      ineigh.data_ptr<int64_t>(), neigh.data_ptr<int64_t>(), height, kngh, nthreads);
  return ineigh;
}

Tensor dwconv_forward_backward(Tensor data, Tensor weight, Tensor neigh) {
  // data: (N, C), weight: (K, 1, C), neigh: (N, K)
  int64_t height = data.size(0);
  int64_t channel = data.size(1);
  int64_t nthreads = height * channel;
  int64_t kngh = neigh.size(1);
  Tensor out = torch::zeros_like(data);
  auto stream = at::cuda::getCurrentCUDAStream();
  dwconv_forward_backward_kernel<float>
      <<<CudaGetBlocks(nthreads), kCudaThreadsNum, 0, stream>>>(
      out.data_ptr<float>(), data.data_ptr<float>(), weight.data_ptr<float>(),
      neigh.data_ptr<int64_t>(), height, channel, kngh, nthreads);
  CUDA_POST_KERNEL_CHECK;
  return out;
}

Tensor dwconv_weight_backward(Tensor grad, Tensor data, Tensor neigh) {
  int64_t height = data.size(0);
  int64_t channel = data.size(1);
  int64_t nthreads = height * channel;
  int64_t kngh = neigh.size(1);
  Tensor out = grad.new_zeros({height / kCudaThreadsNum, kngh, 1, channel});
  auto stream = at::cuda::getCurrentCUDAStream();
  dwconv_weight_backward_kernel<float>
      <<<CudaGetBlocks(nthreads), kCudaThreadsNum, 0, stream>>>(
      out.data_ptr<float>(), grad.data_ptr<float>(), data.data_ptr<float>(),
      neigh.data_ptr<int64_t>(), height, channel, kngh, nthreads);
  CUDA_POST_KERNEL_CHECK;
  return out.sum(0);
}
