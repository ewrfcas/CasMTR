#ifndef _Score_CUDA
#define _Score_CUDA
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> score_cuda_forward(torch::Tensor input1, torch::Tensor input2, torch::Tensor index);
std::vector<at::Tensor> ScoreComputationForward(at::Tensor input1, at::Tensor input2, at::Tensor index);
std::vector<torch::Tensor> score_cuda_backward(torch::Tensor grad_output, torch::Tensor input1, torch::Tensor input2, torch::Tensor index);
std::vector<at::Tensor> ScoreComputationBackward(at::Tensor grad_output, at::Tensor input1, at::Tensor input2, at::Tensor index);
                      
#endif