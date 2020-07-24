#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

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
        
    def count_gradients(self, w_init, w_final_list):
         num_grads = []
         nz_grads = []
         for i in range(0, len(w_final_list)):
            num_grads.append(0)
            nz_grads.append(0)
         for k in w_init.keys():
            for i in range(len(w_final_list)):
                num_grads[i] += w_init[k].numel()
                nz_grads[i] += torch.nonzero((w_init[k] - w_final_list[i][k]) > 0.0001).size(0)
         return num_grads, nz_grads

