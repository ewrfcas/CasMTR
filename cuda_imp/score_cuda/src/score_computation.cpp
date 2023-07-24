#include "score_computation.h"
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// == Forward
std::vector<torch::Tensor> score_cuda_forward(torch::Tensor input1, torch::Tensor input2, torch::Tensor index)
{
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    CHECK_INPUT(index);
    return ScoreComputationForward(input1, input2, index);
}

// == Backward
std::vector<torch::Tensor> score_cuda_backward(torch::Tensor grad_output, torch::Tensor input1, torch::Tensor input2, torch::Tensor index)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    CHECK_INPUT(index);
    return ScoreComputationBackward(grad_output, input1, input2, index);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("score_forward", &score_cuda_forward, "score forward (CUDA)");
  m.def("score_backward", &score_cuda_backward, "score forward (CUDA)");
}