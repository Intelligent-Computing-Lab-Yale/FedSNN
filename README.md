# Federated Learning with Spiking Neural Networks

This repo contains the source code for the paper "Federated Learning with Spiking Neural Networks" (https://arxiv.org/abs/2106.06579).

## Requirements
python>=3.7
pytorch>=1.7.1

## Run

For example, to train a federated SNN model with 10 clients and 2 clients participating in each round:
> python main_fed.py --snn --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.1 --lr_reduce 5 --epochs 100 --local_ep 2 --eval_every 1 --num_users 10 --frac 0.2 --iid --gpu 0 --timesteps 20 --result_dir test

Other options can be found by running
> pythin main_fed.py --help

Sample scripts are provided at `test_cifar10.sh` and `test_cifar100.sh`.

## Ackonwledgements
Initial Code adopted from https://github.com/shaoxiongji/federated-learning

Code for SNN training adopted from https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time



