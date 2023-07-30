#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <vector>
#include "score_computation.h"
#include <stdio.h>

#define ROUND_OFF 50000

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 128
#define MAX_H 8

// #define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// #define GET_BLOCKS(n, t) (n+t-1) / t



template <typename scalar_t>
__global__ void score_computation_forward_kernel(
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> query, // B, N1, dim
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> key, //B, N2, dim
  torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> index, //B, N1, K
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output) //B, N1, K
{
  int b = blockIdx.y;
  int n1 = blockIdx.x;
  int k = threadIdx.x;

  int D = query.size(2);

  int idx = index[b][n1][k];

  for(int ch = 0; ch < D; ch ++){
    output[b][n1][k] += (query[b][n1][ch] * key[b][idx][ch]);
  }
}


std::vector<torch::Tensor> ScoreComputationForward(
  torch::Tensor query, // B, N1, dim
  torch::Tensor key, // B, N2, dim
  torch::Tensor index) // B, N1, K
{
    const auto B = query.size(0);
    const auto N1 = query.size(1);
    const auto K = index.size(2);

    auto output = torch::zeros({B, N1, K},torch::device(torch::kCUDA));

    dim3 totalBlocks(N1, B);
    dim3 threadsPerBlock(K);
    AT_DISPATCH_FLOATING_TYPES(query.type(), "score_computation_forward_kernel", ([&] {
      score_computation_forward_kernel<scalar_t><<<totalBlocks, threadsPerBlock>>>(
          query.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          key.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          index.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
          output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));
  return {output};

}

template <typename scalar_t>
__global__ void score_computation_backward_kernel(
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad, //B, N1, K
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> query, //B, N1, dim
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> key, // B, N2, dim
  torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> index,// B, N1, K
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> query_grad, //B, N1, dim
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> key_grad //B, N2, dim
  ){
  int b = blockIdx.y;
  int n1 = blockIdx.x;
  int k = threadIdx.x;

  int D = query.size(2);

  int idx = index[b][n1][k];
  for(int ch = 0; ch < D; ch++){
    atomicAdd(&query_grad[b][n1][ch], grad[b][n1][k] * key[b][idx][ch]);
    atomicAdd(&key_grad[b][idx][ch], grad[b][n1][k] * query[b][n1][ch]);
  }
}

std::vector<torch::Tensor> ScoreComputationBackward(
  torch::Tensor grad_output, //B, N1, K
  torch::Tensor query, //B, N1, dim
  torch::Tensor key, //B, N2, dim
  torch::Tensor index) //B, N1, K

{

    const auto B = grad_output.size(0);
    const auto N1 = grad_output.size(1);
    const auto N2 = key.size(1);
    const auto K = grad_output.size(2);
    const auto D = key.size(2);


    auto query_grad = torch::zeros({B, N1, D},torch::device(torch::kCUDA));
    auto key_grad = torch::zeros({B, N2, D},torch::device(torch::kCUDA));

    dim3 totalBlocks(N1, B);
    dim3 threadsPerBlock(K);
    
    AT_DISPATCH_FLOATING_TYPES(key.type(), "score_computation_backward_kernel", ([&] {
      score_computation_backward_kernel<scalar_t><<<totalBlocks, threadsPerBlock>>>(
          grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          query.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          key.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          index.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
          query_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
          key_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
          );
    }));

  return {query_grad, key_grad};

}