# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import ocnn
import unittest
from dwconv import octree_dwconv
from dwconv.nn import dwconv_forward_backward, dwconv_weight_backward, inverse_neigh

from .utils import get_batch_octree


class TesOctreeDWConv(unittest.TestCase):

  def test_dwconv_with_conv(self):

    depth = 3
    channel = 2
    octree = get_batch_octree()
    kernel_size = [[3, 3, 3], [3, 1, 1], [1, 3, 1], [1, 1, 3],
                   [2, 2, 2], [3, 3, 1], [1, 3, 3], [3, 1, 3]]

    for i in range(len(kernel_size)):
      for stride in [1]:  # [1, 2]:
        nnum = octree.nnum_nempty[depth]
        rnd_data = torch.randn(nnum, channel)
        ocnn_data = rnd_data.clone().requires_grad_()
        ocnn_dwconv = ocnn.nn.OctreeDWConv(
            channel, kernel_size[i], stride, nempty=True)
        ocnn_out = ocnn_dwconv(ocnn_data, octree, depth)
        ocnn_out.sum().backward()

        data = rnd_data.clone().cuda().requires_grad_()
        weights = ocnn_dwconv.weights.detach().clone().cuda().requires_grad_()
        kernel = ''.join([str(k) for k in kernel_size[i]])
        neigh = octree.get_neigh(depth, kernel, stride, nempty=True).cuda()

        # out = dwconv_forward_backward(data, weights, neigh)
        # grad = torch.full_like(data, fill_value=1)
        # ineigh = inverse_neigh(neigh)
        # grad_d = dwconv_forward_backward(grad, weights, ineigh)

        # # grad_w = dwconv_weight_backward(grad, data, neigh)
        # self.assertTrue(torch.allclose(
        #     out.cpu(), ocnn_out, atol=1e-6))
        # self.assertTrue(torch.allclose(
        #     grad_d.cpu(), ocnn_data.grad, atol=1e-6))

        out = octree_dwconv(data, weights, neigh)
        out.sum().backward()

        self.assertTrue(torch.allclose(
            out.cpu(), ocnn_out, atol=1e-6))
        self.assertTrue(torch.allclose(
            data.grad.cpu(), ocnn_data.grad, atol=1e-6))
        self.assertTrue(torch.allclose(
            weights.grad.cpu(), ocnn_dwconv.weights.grad, atol=5e-5))


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
