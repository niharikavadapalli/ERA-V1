#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:29:37 2023

@author: svaddi
"""

import torch
import numpy as np
from typing import List, Any
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, seed_everything
from utils.dataset import YOLODataset
from utils.dataset_org import YOLODataset_org
import utils.config as config



class YOLODataModule(LightningDataModule):
    def __init__(self,
                 csv_files,
                 img_dir,
                 label_dir,
                 anchors,
                 image_size=416,
                 S=[13, 26, 52],
                 C=20,
                 train_transforms = None,
                 val_transforms = None,
                 test_transforms = None,
                 val_split=0.2,
                 num_workers = 1,
                 pin_memory = False,
                 batch_size = 32):


        super().__init__()
        self.train_csv_path, self.test_csv_path = csv_files
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.anchors = anchors
        self.image_size = image_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.val_split = val_split
        self.S = S
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_dir = config.DATASET
        self.batch_size = batch_size

    def get_dataset_train(self):
        return YOLODataset_org( self.train_csv_path,
                transform = self.train_transforms,
                S=self.S,
                img_dir=self.img_dir,
                label_dir=self.label_dir,
                anchors=self.anchors)

    def get_dataset_test(self):
        return YOLODataset_org(
            self.test_csv_path,
            transform=self.test_transforms,
            S=self.S,
            img_dir=self.img_dir,
            label_dir=self.label_dir,
            anchors=self.anchors)

    def get_dataset_val(self):
        return YOLODataset_org(
            self.train_csv_path,
            transform=self.val_transforms,
            S=self.S,
            img_dir=self.img_dir,
            label_dir=self.label_dir,
            anchors=self.anchors)


    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)
        dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            splits = [train_len, self.val_split]
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return splits


    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == 'fit' or stage is None:
            dataset_train = self.get_dataset_train()
            dataset_val = self.get_dataset_val()

            # Split
            self.train_dataset = self._split_dataset(dataset_train)
            self.val_dataset = self._split_dataset(dataset_val, train=False)

        if stage == 'test' or stage:
            self.test_dataset = self.get_dataset_test()

    def train_dataloader(self):
        train_data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=False)
        self.train_data_loader = train_data_loader
        return train_data_loader

    def val_dataloader(self):
        val_data_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False)
        self.val_data_loader = val_data_loader
        return val_data_loader

    def test_dataloader(self):
        test_data_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False)
        self.test_data_loader = test_data_loader
        return test_data_loader