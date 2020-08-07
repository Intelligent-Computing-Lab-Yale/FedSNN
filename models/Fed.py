#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from typing import Union

def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

class FedLearn(object):
    def __init__(self, args):
        self.args = args

    def FedAvg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg

    def FedAvgSparse(self, w_init, delta_w_locals, sparsity = 90, activity = None, activity_multiplier = None):
        delta_w_avg = {}
        w_avg = {}
        sparse_delta_w_locals = []
        for i in range(0, len(delta_w_locals)):
            sparse_delta_w = {}
            sparse_delta_w_locals.append(sparse_delta_w)
        for k in w_init.keys():
            layer_activity = None
            if "features" in k:
                idx = int(k.split(sep='.')[2])
                layer_activity = activity[idx]
                prev = idx
            elif "classifier" in k:
                idx = int(k.split(sep='.')[2]) + prev + 3
                if idx in activity.keys():
                    layer_activity = activity[idx]
            else:
                print("Unknown Layer!")
            if layer_activity:
                th = percentile(torch.abs(delta_w_locals[0][k]), 100*(1 - layer_activity/activity_multiplier))
            else:
                th = percentile(torch.abs(delta_w_locals[0][k]), sparsity)
            th = torch.FloatTensor([th]).cuda()
            mask = torch.abs(delta_w_locals[0][k]) > th.expand_as(w_init[k])
            sparse_delta_w_locals[0][k] = delta_w_locals[0][k] * mask
            delta_w_avg[k] = (delta_w_locals[0][k] * mask)
        for k in w_init.keys():
            for i in range(1, len(delta_w_locals)):
                th = percentile(torch.abs(delta_w_locals[i][k]), sparsity)
                th = torch.FloatTensor([th]).cuda()
                mask = torch.abs(delta_w_locals[i][k]) > th.expand_as(w_init[k])
                sparse_delta_w_locals[i][k] = delta_w_locals[i][k] * mask
                delta_w_avg[k] += (delta_w_locals[i][k] * mask)
            delta_w_avg[k] = torch.div(delta_w_avg[k], len(delta_w_avg))
            w_avg[k] = w_init[k] + delta_w_avg[k]
        return w_avg, delta_w_avg, sparse_delta_w_locals

    def count_gradients(self, delta_w_locals, sparse_delta_w_locals):
         num_grads = []
         nz_grads = []
         for i in range(0, len(delta_w_locals)):
            num_grads.append(0)
            nz_grads.append(0)
         for k in delta_w_locals[0].keys():
            for i in range(len(delta_w_locals)):
                num_grads[i] += delta_w_locals[i][k].numel()
                nz_grads[i] += torch.nonzero(sparse_delta_w_locals[i][k]).size(0)
         return num_grads, nz_grads

