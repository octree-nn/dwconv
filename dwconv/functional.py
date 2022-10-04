import torch
from torch.autograd import Function

from .nn import dwconv_forward_backward, dwconv_weight_backward, inverse_neigh


class OctreeDWConvFunction(Function):
  r''' Wrap the octree depth-wise convolution with auto-diff.
  '''

  @staticmethod
  def forward(ctx, data: torch.Tensor, weights: torch.Tensor, neigh: torch.Tensor):
    data = data.contiguous()
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


octree_dwconv = OctreeDWConvFunction.apply
