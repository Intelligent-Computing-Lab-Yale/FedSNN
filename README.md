# Federated Learning with Spiking Neural Networks

This repo contains the source code for the paper "Federated Learning with Spiking Neural Networks". 

## Requirements
python>=3.7
pytorch>=1.7.1

## Run

For example, to train a federated SNN model with 100 clients and 10 clients participating in each round:
> python main_fed.py --dataset CIFAR10 --model VGG9 --num_channels 3 --snn --iid --epochs 100 --gpu 0 --lr 0.1 --num_users 100 --frac 0.1

Other options can be found by running
> pythin main_fed.py --help

Sample scripts are provided at `test_cifar10.sh` and `test_cifar100.sh`.

## Ackonwledgements
Initial Code adopted from https://github.com/shaoxiongji/federated-learning

Code for SNN training adopted from https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time



