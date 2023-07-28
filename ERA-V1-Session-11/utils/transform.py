#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 08:25:58 2023

@author: svaddi
"""

from dataset import MyDataset
from torchvision import datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
        A.PadIfNeeded(40,40),
        A.RandomCrop(32,32),
        A.HorizontalFlip(p=0.2),
        A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8, fill_value=(0.49139968, 0.48215841, 0.44653091), mask_fill_value = None),
        A.Normalize(mean= (0.49139968, 0.48215841, 0.44653091), std = (0.24703223, 0.24348513, 0.26158784)),
        ToTensorV2(),
    ])

test_transforms = A.Compose([
        A.Normalize(mean= (0.49139968, 0.48215841, 0.44653091), std = (0.24703223, 0.24348513, 0.26158784)),
        ToTensorV2(),
    ])


def get_loaders(train_transform = train_transforms, test_transform = test_transforms, batch_size=128, use_cuda=True):

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        MyDataset(datasets.CIFAR10('../data', train=True,
                     download=True), transforms=train_transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        MyDataset(datasets.CIFAR10('../data', train=False,
                     download=True), transforms=test_transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader