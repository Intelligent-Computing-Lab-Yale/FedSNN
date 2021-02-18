# Federated Learning with Spiking Neural Networks

## Requirements
python>=3.6  
pytorch>=0.4

## Run

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

NB: for CIFAR-10, `num_channels` must be 3.

## Results
### CIFAR10
### MNIST

## Ackonwledgements
Initial Code adopted from https://github.com/shaoxiongji/federated-learning

Code for SNN training adopted from https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time



## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.

Shaoxiong Ji, Shirui Pan, Guodong Long, Xue Li, Jing Jiang, and Zi Huang. Learning private neural language modeling with attentive aggregation. In the 2019 International Joint Conference on Neural Networks (IJCNN), 2019. [[Paper](https://arxiv.org/abs/1812.07108)] [[Code](https://github.com/shaoxiongji/fed-att)]

Jing Jiang, Shaoxiong Ji, and Guodong Long. Decentralized knowledge acquisition for mobile internet applications. World Wide Web, 2020. [[Paper](https://link.springer.com/article/10.1007/s11280-019-00775-w)]


