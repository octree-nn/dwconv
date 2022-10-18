import torch
import ocnn
from typing import List
from ocnn.octree import Octree
from torch.autograd import Function

from . import core
from .core import (
    dwconv_forward_backward, dwconv_weight_backward, inverse_neigh,
    dwconv_forward_backward_t, dwconv_weight_backward_t)


class OctreeDWConvFunction(Function):
  r''' Wrap the octree depth-wise convolution with auto-diff.
  '''

  @staticmethod
  def forward(ctx, data: torch.Tensor, weights: torch.Tensor, neigh: torch.Tensor):
    data = data.contiguous()
    weights = weights.contiguous()
    neigh = neigh.contiguous()
    out = dwconv_forward_backward(data, weights, neigh)
    ctx.save_for_backward(data, weights, neigh)
    return out

  @staticmethod
  def backward(ctx, grad):
    data, weights, neigh = ctx.saved_tensors
    grad = grad.contiguous()

    grad_d = None
    if ctx.needs_input_grad[0]:
      ineigh = inverse_neigh(neigh)
      grad_d = dwconv_forward_backward(grad, weights, ineigh)

    grad_w = None
    if ctx.needs_input_grad[1]:
      grad_w = dwconv_weight_backward(grad, data, neigh)
    return grad_d, grad_w, None


class OctreeDWConvTFunction(Function):
  r''' Wrap the octree depth-wise convolution with auto-diff.
  '''

  @staticmethod
  def forward(ctx, data: torch.Tensor, weights: torch.Tensor, neigh: torch.Tensor):
    data = data.contiguous()
    weights = weights.contiguous()
    neigh = neigh.contiguous()
    out = dwconv_forward_backward_t(data, weights, neigh)
    ctx.save_for_backward(data, weights, neigh)
    return out

  @staticmethod
  def backward(ctx, grad):
    data, weights, neigh = ctx.saved_tensors
    grad = grad.contiguous()

    grad_d = None
    if ctx.needs_input_grad[0]:
      ineigh = inverse_neigh(neigh)
      grad_d = dwconv_forward_backward_t(grad, weights, ineigh)

    grad_w = None
    if ctx.needs_input_grad[1]:
      grad_w = dwconv_weight_backward_t(grad, data, neigh)
    return grad_d, grad_w, None


octree_dwconv = OctreeDWConvFunction.apply
octree_dwconv_t = OctreeDWConvTFunction.apply


class OctreeDWConv(ocnn.nn.OctreeDWConv):
  r''' Speeds up `ocnn.nn.OctreeDWConv` with CUDA.
  '''

  def __init__(self, channels: int, kernel_size: List[int] = [3],
               nempty: bool = False, use_bias: bool = False):
    super().__init__(in_channels=channels, kernel_size=kernel_size, stride=1,
                     nempty=nempty, use_bias=use_bias)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    neigh = octree.get_neigh(depth, self.kernel, self.stride, self.nempty)
    out = octree_dwconv(data, self.weights, neigh)
    if self.use_bias:
      out += self.bias
    return out


class OctreeDWConvT(ocnn.nn.OctreeDWConv):
  r''' Speeds up `ocnn.nn.OctreeDWConv` with CUDA.
  '''

  def __init__(self, channels: int, kernel_size: List[int] = [3],
               nempty: bool = False, use_bias: bool = False):
    super().__init__(in_channels=channels, kernel_size=kernel_size, stride=1,
                     nempty=nempty, use_bias=use_bias)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    data = data.t()
    weights = self.weights.squeeze().t()
    neigh = octree.get_neigh(depth, self.kernel, self.stride, self.nempty)
    out = octree_dwconv_t(data, weights, neigh)
    out = out.t()
    if self.use_bias:
      out += self.bias
    return out
