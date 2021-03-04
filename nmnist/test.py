import os
import pysnn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pysnn.datasets import nmnist_train_test

root = "data"
if os.path.isdir(root):
    train_dataset, test_dataset = nmnist_train_test(root)
else:
    raise NotADirectoryError(
        "Make sure to download the N-MNIST dataset from https://www.garrickorchard.com/datasets/n-mnist and put it in the 'nmnist' folder."
    )
batch_size = 32
num_workers = 2
print(train_dataset.size())
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)