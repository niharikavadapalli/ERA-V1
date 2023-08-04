#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:29:37 2023

@author: svaddi
"""

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision import transforms

class CustomCifar10DataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers = 2):
        # Initialize the class. Set up the datadir, image dims, and num classes
        super().__init__()
        self.BATCH_SIZE = batch_size
        self.NUM_WORKERS = num_workers
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.491,0.482,0.447),std=(0.247,0.244,0.262))
            ]
        )
        
        self.test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.491,0.482,0.447),std=(0.247,0.244,0.262))
            ]
        )
        self.val_transforms = self.test_transforms
        self.prepare_data()
        self.setup()
        
    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.train_transforms)
            self.train_dataset, self.val_dataset = random_split(cifar10_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)