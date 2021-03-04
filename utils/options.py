#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=16, help="test batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--lr_interval', default='0.60 0.80 0.90', type=str, help='intervals at which to reduce lr, expressed as %%age of total epochs')

    parser.add_argument('--lr_reduce', default=5, type=int, help='reduction factor for learning rate')
    parser.add_argument('--timesteps', default=50, type=int, help='simulation timesteps')
    parser.add_argument('--leak', default=1.0, type=float, help='membrane leak')
    parser.add_argument('--scaling_factor', default=0.7, type=float, help='scaling factor for thresholds at reduced timesteps')
    parser.add_argument('--default_threshold', default=1.0, type=float, help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation', default='Linear', type=str, help='SNN activation function', choices=['Linear', 'STDB'])
    parser.add_argument('--alpha', default=0.3, type=float, help='parameter alpha for STDB')
    parser.add_argument('--beta', default=0.01, type=float, help='parameter beta for STDB')
    parser.add_argument('--snn_kernel_size', default=3, type=int, help='filter size for the conv layers')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay parameter for the optimizer')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout percentage for conv layers')

    parser.add_argument('--momentum', type=float, default=0.95, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--eval_every', type=int, default=10, help='Frequency of model evaluation')
    parser.add_argument('--pretrained_model', type=str, default=None, help="Path for the pre-trained mode if any")
    parser.add_argument('--result_dir', type=str, default="results", help="Directory to store results")
    parser.add_argument('--snn', action='store_true', help="Whether to train SNN or ANN")
    parser.add_argument('--train_acc_batches', default=200, type=int, help='print training progress after this many batches')
    parser.add_argument('--straggler_prob', type=float, default=0.0, help="straggler probability")
    parser.add_argument('--grad_noise_stdev', type=float, default=0.0, help="Noise level for gradients")
    parser.add_argument('--dvs', action='store_true', help="Whether the input data is DVS")
    args = parser.parse_args()
    return args
