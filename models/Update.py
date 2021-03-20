#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import sys
import os
sys.path.append(os.getcwd() + '/..')
from ddd20.hdf5_deeplearn_utils import MultiHDF5VisualIteratorFederated


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)

    def train(self, net):
        net.train()
        # train and update
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay, amsgrad = True)
        else:
            print("Invalid optimizer")

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                if self.args.verbose and (batch_idx + 1) % self.args.train_acc_batches == 0:
                    thresholds = []
                    for value in net.module.threshold.values():
                        thresholds = thresholds + [round(value.item(), 2)]
                    print('Epoch: {}, batch {}, threshold {}, leak {}, timesteps {}'.format(iter, batch_idx + 1, thresholds, net.module.leak.item(), net.module.timesteps))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdateDDD(object):
    def __init__(self, args, dataset_keys=None, h5fs=None, client_id = 0):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.dataset_keys = dataset_keys
        self.h5fs = h5fs
        self.client_id = client_id
        self.ldr_train = MultiHDF5VisualIteratorFederated()

    def train(self, net):
        net.train()
        # train and update
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay, amsgrad = True)
        else:
            print("Invalid optimizer")

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train.flow(self.h5fs, self.dataset_keys, 'train_idxs', batch_size=self.args.bs, shuffle=True, iid = self.args.iid, client_id = self.client_id, num_clients = self.args.num_users)):
                images, labels = torch.from_numpy(images).to(self.args.device), torch.from_numpy(labels).to(self.args.device)
                net.zero_grad()
                output = net(images)
                loss = self.loss_func(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)