import torch
from torchvision import datasets, transforms
from src.sfl.data_loader import CachedDataset, get_cifar10_transforms
raw = datasets.CIFAR10('./data', train=True, download=True, transform=get_cifar10_transforms(True))
print('Loaded raw.')
d = CachedDataset(raw)
print('Done.')
