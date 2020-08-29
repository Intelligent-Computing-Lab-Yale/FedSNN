#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
import torch
import torch.nn as nn

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, VGG11_CIFAR100, VGG
from models.Fed import FedLearn
from models.test import test_img
import pytorch_cifar.models as pcm
import hybrid_snn_conversion.self_models as snn_models

class SubsetLoaderMNIST(datasets.MNIST):
    def __init__(self, *args, exclude_list=[], **kwargs):
        super(SubsetLoaderMNIST, self).__init__(*args, **kwargs)

        if exclude_list == []:
            return

        labels = np.array(self.targets)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask]

class SubsetLoaderCIFAR10(datasets.CIFAR10):
    def __init__(self, *args, exclude_list=[], **kwargs):
        super(SubsetLoaderCIFAR10, self).__init__(*args, **kwargs)

        if exclude_list == []:
            return

        labels = np.array(self.targets)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask]

class SubsetLoaderCIFAR100(datasets.CIFAR100):
    def __init__(self, *args, exclude_list=[], **kwargs):
        super(SubsetLoaderCIFAR100, self).__init__(*args, **kwargs)

        if exclude_list == []:
            return

        labels = np.array(self.targets)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask]

def partition_dataset(dataset, part = "full"):
    if part == "full":
        return dataset
    elif part == "first":
        dataset.data = dataset.data[0::2]
        dataset.targets = dataset.targets[0::2]
    elif part == "second":
        dataset.data = dataset.data[1::2]
        dataset.targets = dataset.targets[1::2]
    return dataset

def find_activity(batch_size=512, timesteps=2500, architecture='VGG5', num_batches = 10):
    loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    activity = []
    activity_mask = []
    pos=0
    
    def find(layer, pos):
        print('Finding activity for layer {}'.format(layer))
        tot_spikes = 0.0
        nz_spikes = 0.0
        tot_activity_mask = None
        for batch_idx, (data, target) in enumerate(loader):
            
            data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                net_glob.eval()
                tot_spikes_, nz_spikes_, activity_mask_ = net_glob(data, find_activity=True, activity_layer=layer)
                tot_spikes += tot_spikes_
                nz_spikes += nz_spikes_
                if tot_activity_mask == None:
                    tot_activity_mask = activity_mask_
                else:
                    tot_activity_mask += activity_mask_
                if batch_idx==(num_batches - 1):
                    activity.append(nz_spikes/tot_spikes)
                    activity_mask.append(tot_activity_mask)
                    pos = pos+1
                    print(' {}'.format(activity))
                    break
        return pos

    if architecture.lower().startswith('vgg'):                                     
        for l in net_glob.features.named_children():                           
            if isinstance(l[1], nn.Conv2d):
                pos = find(int(l[0]), pos)
            
        for c in net_glob.classifier.named_children():                         
            if isinstance(c[1], nn.Linear):                                        
                if (int(l[0])+int(c[0])+1) == (len(net_glob.features) + len(net_glob.classifier) -1):       
                    pass
                else:
                    pos = find(int(l[0])+int(c[0])+1, pos)                         
                    
    if architecture.lower().startswith('res'):                                     
        for l in net_glob.pre_process.named_children():                        
            if isinstance(l[1], nn.Conv2d):                                        
                pos = find(int(l[0]), pos)
    print('Spike activity: {}'.format(activity))
    return activity, activity_mask


def find_threshold(batch_size=512, timesteps=2500, architecture='VGG16'):
    loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    net_glob.network_update(timesteps=args.timesteps, leak=1.0)

    pos=0
    thresholds=[]
    
    def find(layer, pos):
        max_act=0
        
        print('Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            
            data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                net_glob.eval()
                output = net_glob(data, find_max_mem=True, max_mem_layer=layer)
                if output>max_act:
                    max_act = output.item()

                #f.write('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                if batch_idx==0:
                    thresholds.append(max_act)
                    pos = pos+1
                    print(' {}'.format(thresholds))
                    net_glob.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                    break
        return pos

    if architecture.lower().startswith('vgg'):                                     
        for l in net_glob.features.named_children():                           
            if isinstance(l[1], nn.Conv2d):
                pos = find(int(l[0]), pos)
            
        for c in net_glob.classifier.named_children():                         
            if isinstance(c[1], nn.Linear):                                        
                if (int(l[0])+int(c[0])+1) == (len(net_glob.features) + len(net_glob.classifier) -1):       
                    pass
                else:
                    pos = find(int(l[0])+int(c[0])+1, pos)                         
                    
    if architecture.lower().startswith('res'):                                     
        for l in net_glob.pre_process.named_children():                        
            if isinstance(l[1], nn.Conv2d):                                        
                pos = find(int(l[0]), pos)
    print('ANN thresholds: {}'.format(thresholds))
    return thresholds


if __name__ == '__main__':
    # parse args
    args = args_parser()
    torch.manual_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    exclude_list = []
    if args.subset == "odd":
        exclude_list = list(range(0, args.num_classes,2))
    elif args.subset == "even":
        exclude_list = list(range(1, args.num_classes,2))

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = SubsetLoaderMNIST('../data/mnist/', train=True, download=True, transform=trans_mnist, exclude_list=exclude_list)
        dataset_test = SubsetLoaderMNIST('../data/mnist/', train=False, download=True, transform=trans_mnist, exclude_list=exclude_list)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'CIFAR10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = SubsetLoaderCIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar, exclude_list=exclude_list)
        dataset_test = SubsetLoaderCIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar, exclude_list=exclude_list)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'CIFAR100':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = SubsetLoaderCIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
        dataset_test = SubsetLoaderCIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    model_args = {'args': args}
    if args.snn:
        if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
            if args.model[0:3].lower() == 'vgg':
                model_args = {'vgg_name': args.model, 'activation': args.activation, 'labels': args.num_classes, 'timesteps': args.timesteps, 'leak': args.leak, 'default_threshold': args.default_threshold, 'alpha': args.alpha, 'beta': args.beta, 'dropout': args.dropout, 'kernel_size': args.snn_kernel_size, 'dataset': args.dataset}
                net_glob = snn_models.VGG_SNN_STDB(**model_args).cuda()
    elif (args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100') and args.model[0:3].lower() == 'vgg':
        model_args = {'vgg_name': args.model, 'labels': args.num_classes, 'dataset': args.dataset, 'kernel_size': args.snn_kernel_size, 'dropout': args.dropout}
        net_glob = snn_models.VGG(**model_args).cuda()
    elif args.dataset == 'CIFAR10':
        if args.model == 'MobileNetV2':
            net_glob = pcm.MobileNetV2().to(args.device)
        else:
            exit("Invalid model")
    elif args.model == 'cnn' and (args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100'):
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'VGG' and args.dataset == 'CIFAR10':
        net_glob = VGG(args=args).to(args.device)
    elif args.model == 'vgg11' and args.dataset == 'CIFAR100':
        net_glob = VGG11_CIFAR100(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        model_args = {'dim_in': len_in, 'dim_hidden': 200, 'dim_out': args.num_classes}
        net_glob = MLP(**model_args ).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # copy weights
    if args.pretrained_model:
        net_glob.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
        if args.snn:
            thresholds = find_threshold(batch_size=512, timesteps=1000, architecture = args.model)
            net_glob.threshold_update(scaling_factor = args.scaling_factor, thresholds = thresholds[:])
            activity, activity_mask = find_activity(batch_size=512, timesteps=1000, architecture = args.model, num_batches = 20)
            net_glob.activity_update(activity = activity[:])

    net_glob = nn.DataParallel(net_glob)
    # training
    loss_train_list = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # metrics to store
    ms_acc_train_list, ms_loss_train_list = [], []
    ms_acc_test_list, ms_loss_test_list = [], []
    ms_num_client_list, ms_tot_comm_cost_list, ms_avg_comm_cost_list, ms_max_comm_cost_list = [], [], [], []
    ms_tot_nz_grad_list, ms_avg_nz_grad_list, ms_max_nz_grad_list = [], [], []

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Initial Training accuracy: {:.2f}".format(acc_train))
    print("Initial Testing accuracy: {:.2f}".format(acc_test))

    # Add metrics to store
    ms_acc_train_list.append(acc_train)
    ms_acc_test_list.append(acc_test)
    ms_loss_train_list.append(loss_train)
    ms_loss_test_list.append(loss_test)

    # Define LR Schedule
    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))

    # Define Fed Learn object
    fl = FedLearn(args)

    for iter in range(args.epochs):
        net_glob.train()
        if args.snn:
            net_glob.module.network_update(timesteps=args.timesteps, leak=args.leak)
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            print(type(net_glob.module))
            model_copy = type(net_glob.module)(**model_args) # get a new instance
            if args.snn:
                thresholds = []
                for value in net_glob.module.threshold.values():
                    thresholds = thresholds + [value.item()]
                model_copy.threshold_update(scaling_factor=1.0, thresholds=thresholds)
            model_copy = nn.DataParallel(model_copy)
            model_copy.load_state_dict(net_glob.state_dict()) # copy weights and stuff
            w, loss = local.train(net=model_copy.to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        # w_glob = fl.FedAvg(w_locals)
        w_init = net_glob.state_dict()
        delta_w_locals = []
        for i in range(0, len(w_locals)):
            delta_w = {}
            for k in w_init.keys():
                delta_w[k] = w_locals[i][k] - w_init[k]
            delta_w_locals.append(delta_w)
        if args.snn:
            activity = net_glob.module.activity
        else:
            activity = None
            activity_mask = None
        w_glob, delta_w_avg, sparse_delta_w_locals = fl.FedAvgSparse(w_init, delta_w_locals, th_basis = args.sparsity_basis, pruning_type = args.pruning_type, sparsity = args.grad_sparsity, activity = activity, activity_multiplier = args.activity_multiplier, activity_mask = activity_mask)
 
        comm_cost, nz_grad = fl.count_gradients(delta_w_locals, sparse_delta_w_locals)
 
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
 
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_list.append(loss_avg)
 
        if iter % args.eval_every == 0:
            # testing
            net_glob.eval()
            acc_train, loss_train = test_img(net_glob, dataset_train, args)
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Round {:d}, Training accuracy: {:.2f}".format(iter, acc_train))
            print("Round {:d}, Testing accuracy: {:.2f}".format(iter, acc_test))
 
            # Add metrics to store
            ms_acc_train_list.append(acc_train)
            ms_acc_test_list.append(acc_test)
            ms_loss_train_list.append(loss_train)
            ms_loss_test_list.append(loss_test)
 
            # print communication cost
            tot_comm_cost = sum(comm_cost)
            avg_comm_cost = sum(comm_cost) / len(comm_cost)
            max_comm_cost = max(comm_cost)
            print('Round {:3d}, Num Clients {}, Tot. Comm. Cost {:.1f}, Average Comm. Cost {:.1f}, Max Comm. Cost {:.1f}'.format(iter, len(comm_cost), tot_comm_cost, avg_comm_cost, max_comm_cost))
            ms_num_client_list.append(len(comm_cost))
            ms_tot_comm_cost_list.append(tot_comm_cost)
            ms_avg_comm_cost_list.append(avg_comm_cost)
            ms_max_comm_cost_list.append(max_comm_cost)
 
            tot_nz_grad = sum(nz_grad)
            avg_nz_grad = sum(nz_grad) / len(nz_grad)
            max_nz_grad = max(nz_grad)
            print('Round {:3d}, Num Clients {}, Tot. Comm. Cost {:.1f}, Average Comm. Cost {:.1f}, Max Comm. Cost {:.1f}'.format(iter, len(nz_grad), tot_nz_grad, avg_nz_grad, max_nz_grad))
            ms_tot_nz_grad_list.append(tot_nz_grad)
            ms_avg_nz_grad_list.append(avg_nz_grad)
            ms_max_nz_grad_list.append(max_nz_grad)
 
        if iter in lr_interval:
            args.lr = args.lr/args.lr_reduce

    Path('./{}'.format(args.result_dir)).mkdir(parents=True, exist_ok=True)
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train_list)), loss_train_list)
    plt.ylabel('train_loss')
    plt.savefig('./{}/fed_loss_{}_{}_{}_{}_C{}_iid{}.png'.format(args.result_dir,args.dataset, args.subset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Final Training accuracy: {:.2f}".format(acc_train))
    print("Final Testing accuracy: {:.2f}".format(acc_test))

    # Add metrics to store
    ms_acc_train_list.append(acc_train)
    ms_acc_test_list.append(acc_test)
    ms_loss_train_list.append(loss_train)
    ms_loss_test_list.append(loss_test)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(ms_acc_train_list)), ms_acc_train_list)
    plt.plot(range(len(ms_acc_test_list)), ms_acc_test_list)
    plt.plot()
    plt.ylabel('Accuracy')
    plt.legend(['Training acc', 'Testing acc'])
    plt.savefig('./{}/fed_acc_{}_{}_{}_{}_C{}_iid{}.png'.format(args.result_dir, args.dataset, args.subset, args.model, args.epochs, args.frac, args.iid))

    # Write metric store into a CSV
    metrics_df = pd.DataFrame(
        {
            'Train acc': ms_acc_train_list,
            'Test acc': ms_acc_test_list,
            'Train loss': ms_loss_train_list,
            'Test loss': ms_loss_test_list
        })
    metrics_df.to_csv('./{}/fed_stats_{}_{}_{}_{}_C{}_iid{}.csv'.format(args.result_dir, args.dataset, args.subset, args.model, args.epochs, args.frac, args.iid), sep='\t')

    comm_metrics_df = pd.DataFrame(
        {
            'num_clients': ms_num_client_list,
            'tot_comm_cost': ms_tot_comm_cost_list,
            'avg_comm_cost': ms_avg_comm_cost_list,
            'max_comm_cost': ms_max_comm_cost_list,
            'tot_nz_grad': ms_tot_nz_grad_list,
            'avg_nz_grad': ms_avg_nz_grad_list,
            'max_nz_grad': ms_max_nz_grad_list
        })
    comm_metrics_df.to_csv('./{}/fed_comm_stats_{}_{}_{}_{}_C{}_iid{}.csv'.format(args.result_dir, args.dataset, args.subset, args.model, args.epochs, args.frac, args.iid), sep='\t')

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in net_glob.state_dict():
        print(param_tensor, "\t", net_glob.state_dict()[param_tensor].size())

    torch.save(net_glob.module.state_dict(), './{}/saved_model'.format(args.result_dir))
