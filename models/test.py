#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.getcwd() + '/..')
from ddd20.hdf5_deeplearn_utils import MultiHDF5VisualIteratorFederated


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy.item(), test_loss

def test_img_ddd(net_g, args, h5fs, dataset_keys, subset):
    net_g.eval()
    # testing
    test_loss = 0
    data_loader = MultiHDF5VisualIteratorFederated()
    loss_func = nn.MSELoss()
    count = 0
    eva = 0
    client_ids = [i for i in range(args.num_users)]
    for client_id in client_ids:
        for idx, (data, target) in enumerate(data_loader.flow(h5fs, dataset_keys, subset + '_idxs', batch_size=args.bs, shuffle=True, iid = args.iid, client_id = client_id, num_clients = 1)):
            if args.gpu != -1:
                data, target = torch.from_numpy(data).to(args.device), torch.from_numpy(target).to(args.device)
            output = net_g(data)
            # sum up batch loss
            test_loss += loss_func(output, target).item()
            eva += (1 - torch.div(torch.var(output - target),torch.var(target)).item())
            if count%200 == 0:
                print(output[0], target[0], test_loss)
            count += 1 # Count number of batches

    test_loss /= (count)
    eva /= count
    test_loss = test_loss**(0.5)
    return test_loss, eva

def comp_activity(net_g, dataset, args):
    net_g.eval()
    # testing
    data_loader = DataLoader(dataset, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        activity = torch.zeros(net_g(data, count_active_layers = True))
        break
    batch_count = 0
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        activity += torch.tensor(net_g(data, report_activity = True))
        # sum up batch loss
        batch_count += 1
    activity = activity/batch_count

    return activity