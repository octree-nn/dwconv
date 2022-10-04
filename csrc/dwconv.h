#pragma once
#include <torch/extension.h>

using torch::Tensor;

Tensor dwconv_forward_backward(Tensor data, Tensor weight, Tensor neigh);
Tensor dwconv_weight_backward(Tensor grad, Tensor data, Tensor neigh);
Tensor inverse_neigh(Tensor neigh);
