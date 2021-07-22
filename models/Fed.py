#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import random
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

def model_diff(w, w_init):
    diff = 0
    for k in w_init.keys():
        if not ("num_batches_tracked" in k):
            diff += torch.linalg.norm(w[k] - w_init[k])
    return diff

def model_deviation(w_locals, w_init):
    model_deviation_list = []
    print("Num clients:",len(w_locals))
    for w in w_locals:
        model_deviation_list.append(model_diff(w, w_init).item())
    return model_deviation_list

class FedLearn(object):
    def __init__(self, args):
        self.args = args

    def FedAvg(self, w, w_init = None):
        non_stragglers = [1]*len(w)
        for i in range(1, len(w)):
            epsilon = random.uniform(0, 1)
            if epsilon < self.args.straggler_prob:
                non_stragglers[i] = 0
        w_avg = copy.deepcopy(w[0]) 
        for k in w_avg.keys():
            if w_init:
                w_avg[k] = w_avg[k] + torch.mean(torch.abs(w_init[k] - w[0][k])*1.0) * torch.randn(w[0][k].size()) * self.args.grad_noise_stdev # Scale the noise by mean of the absolute value of the model updates
            else:
                w_avg[k] = w_avg[k] + torch.randn(w[0][k].size()) * self.args.grad_noise_stdev # Add gaussian noise to the model updates
        for k in w_avg.keys():
            for i in range(1, len(w)):
                if non_stragglers[i] == 1:
                    if w_init:
                        w_avg[k] = w_avg[k] + w[i][k] + torch.mean(torch.abs(w_init[k] - w[i][k])*1.0) * torch.randn(w[i][k].size()) * self.args.grad_noise_stdev # Scale the noise by mean of the absolute value of the model updates
                        # 1.0 is to convert into float
                    else:
                        w_avg[k] = w_avg[k] + w[i][k] + torch.randn(w[i][k].size()) * self.args.grad_noise_stdev # Add gaussian noise to the model updates
            w_avg[k] = torch.div(w_avg[k], sum(non_stragglers))
        return w_avg

    def FedAvgSparse(self, w_init, delta_w_locals, th_basis = "magnitude", pruning_type = "uniform", sparsity = 0, activity = None, activity_multiplier = 1, activity_mask = None):
        # th_basis -> on what basis the threshold is calculated - magnitude or activity
        # pruningy_type -> uniform or dynamic: uniform will have equal sparsity among all layers. dynamic has different sparsity for different layers based on activity.
        # Only magnitude based uniform is applicable for ANNs
        delta_w_avg = {}
        w_avg = {}
        sparse_delta_w_locals = []
        for i in range(0, len(delta_w_locals)):
            sparse_delta_w = {}
            sparse_delta_w_locals.append(sparse_delta_w)
        for k in w_init.keys():
            # Threshold Calculation
            if th_basis == "magnitude" and pruning_type == "uniform":
                th = percentile(torch.abs(delta_w_locals[0][k]), sparsity)
                th = torch.FloatTensor([th]).cuda()
                mask = torch.abs(delta_w_locals[0][k]) > th.expand_as(w_init[k])
            elif th_basis == "magnitude" and pruning_type == "dynamic":
                if activity is None:
                    print("Layer activity not available. Dynamic sparsity not possible")
                if "features" in k:
                    idx = int(k.split(sep='.')[2])
                    layer_activity = activity[idx]
                    prev = idx
                elif "classifier" in k:
                    idx = int(k.split(sep='.')[2]) + prev + 3
                    if idx in activity.keys():
                        layer_activity = activity[idx]
                    else:
                        layer_activity = sum(activity) / len(activity)
                else:
                    print("Unknown Layer!")
                s = 100*(1 - layer_activity/activity_multiplier)
                print("sparsity", s)
                th = percentile(torch.abs(delta_w_locals[0][k]), s)
                print("Threshold", th)
                th = torch.FloatTensor([th]).cuda()
                mask = torch.abs(delta_w_locals[0][k]) > th.expand_as(w_init[k])
            elif th_basis == "activity" and pruning_type == "uniform":
                if activity_mask is None:
                    print("Activity mask is not available. Activity based pruning not possible")
                if "features" in k:
                    idx = int(k.split(sep='.')[2])
                    layer_activity_mask = activity_mask[idx]
                    prev = idx
                elif "classifier" in k:
                    idx = int(k.split(sep='.')[2]) + prev + 3
                    if idx in activity_mask.keys():
                        layer_activity_mask = activity_mask[idx]
                    else:
                        layer_activity_mask = torch.tensor(1)
                else:
                    print("Unknown Layer!")
                if layer_activity_mask.shape == torch.Size([]):
                    th = percentile(torch.abs(delta_w_locals[0][k]), sparsity)
                    th = torch.FloatTensor([th]).cuda()
                    mask = torch.abs(delta_w_locals[0][k]) > th.expand_as(w_init[k])
                else:
                    th = percentile(layer_activity_mask, sparsity)
                    th = torch.FloatTensor([th]).cuda()
                    mask = layer_activity_mask > th.expand_as(w_init[k])
            elif th_basis == "activity" and pruning_type == "dynamic":
                if activity is None:
                    print("Layer activity not available. Dynamic sparsity not possible")
                if activity_mask is None:
                    print("Activity mask is not available. Activity based pruning not possible")
                if "features" in k:
                    idx = int(k.split(sep='.')[2])
                    layer_activity = activity[idx]
                    layer_activity_mask = activity_mask[idx]
                    prev = idx
                elif "classifier" in k:
                    idx = int(k.split(sep='.')[2]) + prev + 3
                    if idx in activity.keys():
                        layer_activity = activity[idx]
                    else:
                        layer_activity = sum(activity) / len(activity)
                    if idx in activity_mask.keys():
                        layer_activity_mask = activity_mask[idx]
                    else:
                        layer_activity_mask = torch.tensor(1)
                else:
                    print("Unknown Layer!")
                s = 100*(1 - layer_activity/activity_multiplier)
                if layer_activity_mask.shape == torch.Size([]):
                    th = percentile(torch.abs(delta_w_locals[0][k]), s)
                    th = torch.FloatTensor([th]).cuda()
                    mask = torch.abs(delta_w_locals[0][k]) > th.expand_as(w_init[k])
                else:
                    th = percentile(layer_activity_mask, s)
                    th = torch.FloatTensor([th]).cuda()
                    mask = layer_activity_mask > th.expand_as(w_init[k])
            else:
                print("Unknown threshold basis or pruning_type. Available options: th_basis - magnitude or activity, pruning_type - uniform or dynamic")
            sparse_delta_w_locals[0][k] = delta_w_locals[0][k] * mask
            delta_w_avg[k] = (delta_w_locals[0][k] * mask)
        for k in w_init.keys():
            for i in range(1, len(delta_w_locals)):
                # Threshold Calculation
                if th_basis == "magnitude" and pruning_type == "uniform":
                    th = percentile(torch.abs(delta_w_locals[i][k]), sparsity)
                    th = torch.FloatTensor([th]).cuda()
                    mask = torch.abs(delta_w_locals[i][k]) > th.expand_as(w_init[k])
                elif th_basis == "magnitude" and pruning_type == "dynamic":
                    if activity is None:
                        print("Layer activity not available. Dynamic sparsity not possible")
                    if "features" in k:
                        idx = int(k.split(sep='.')[2])
                        layer_activity = activity[idx]
                        prev = idx
                    elif "classifier" in k:
                        idx = int(k.split(sep='.')[2]) + prev + 3
                        if idx in activity.keys():
                            layer_activity = activity[idx]
                        else:
                            layer_activity = sum(activity) / len(activity)
                    else:
                        print("Unknown Layer!")
                    s = 100*(1 - layer_activity/activity_multiplier)
                    print("sparsity", s)
                    th = percentile(torch.abs(delta_w_locals[i][k]), s)
                    print("Threshold", th)
                    th = torch.FloatTensor([th]).cuda()
                    mask = torch.abs(delta_w_locals[i][k]) > th.expand_as(w_init[k])
                elif th_basis == "activity" and pruning_type == "uniform":
                    if activity_mask is None:
                        print("Activity mask is not available. Activity based pruning not possible")
                    if "features" in k:
                        idx = int(k.split(sep='.')[2])
                        layer_activity_mask = activity_mask[idx]
                        prev = idx
                    elif "classifier" in k:
                        idx = int(k.split(sep='.')[2]) + prev + 3
                        if idx in activity_mask.keys():
                            layer_activity_mask = activity_mask[idx]
                        else:
                            layer_activity_mask = torch.tensor(1)
                    else:
                        print("Unknown Layer!")
                    if layer_activity_mask.shape == torch.Size([]):
                        th = percentile(torch.abs(delta_w_locals[i][k]), sparsity)
                        th = torch.FloatTensor([th]).cuda()
                        mask = torch.abs(delta_w_locals[i][k]) > th.expand_as(w_init[k])
                    else:
                        th = percentile(layer_activity_mask, sparsity)
                        th = torch.FloatTensor([th]).cuda()
                        mask = layer_activity_mask > th.expand_as(w_init[k])
                elif th_basis == "activity" and pruning_type == "dynamic":
                    if activity is None:
                        print("Layer activity not available. Dynamic sparsity not possible")
                    if activity_mask is None:
                        print("Activity mask is not available. Activity based pruning not possible")
                    if "features" in k:
                        idx = int(k.split(sep='.')[2])
                        layer_activity = activity[idx]
                        layer_activity_mask = activity_mask[idx]
                        prev = idx
                    elif "classifier" in k:
                        idx = int(k.split(sep='.')[2]) + prev + 3
                        if idx in activity.keys():
                            layer_activity = activity[idx]
                        else:
                            layer_activity = sum(activity) / len(activity)
                        if idx in activity_mask.keys():
                            layer_activity_mask = activity_mask[idx]
                        else:
                            layer_activity_mask = torch.tensor(1)
                    else:
                        print("Unknown Layer!")
                    s = 100*(1 - layer_activity/activity_multiplier)
                    if layer_activity_mask.shape == torch.Size([]):
                        th = percentile(torch.abs(delta_w_locals[i][k]), s)
                        th = torch.FloatTensor([th]).cuda()
                        mask = torch.abs(delta_w_locals[i][k]) > th.expand_as(w_init[k])
                    else:
                        th = percentile(layer_activity_mask, s)
                        th = torch.FloatTensor([th]).cuda()
                        mask = layer_activity_mask > th.expand_as(w_init[k])
                else:
                    print("Unknown threshold basis or pruning_type. Available options: th_basis - magnitude or activity, pruning_type - uniform or dynamic")
                sparse_delta_w_locals[i][k] = delta_w_locals[i][k] * mask
                delta_w_avg[k] = (delta_w_locals[i][k] * mask)
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