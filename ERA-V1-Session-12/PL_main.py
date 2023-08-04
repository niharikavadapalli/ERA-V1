#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 08:56:19 2023

@author: svaddi
"""

from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models.PL_custom_resnet_model import LitResnet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torch_lr_finder import LRFinder


def get_data_module(PATH_DATASETS,BATCH_SIZE,NUM_WORKERS):
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    
    cifar10_dm = CIFAR10DataModule(
        data_dir=PATH_DATASETS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    return cifar10_dm

def create_and_train_pl_model(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, datamodule, best_lr):
    model = LitResnet(lr=0.01, BATCH_SIZE=BATCH_SIZE, best_lr=best_lr)

    trainer = Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)
    return trainer, model

def get_max_lr(datamodule,BATCH_SIZE,NUM_WORKERS, device):
    lr = 0.01
    dummy_model = LitResnet(lr=lr)
    optimizer = torch.optim.SGD(
        dummy_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    loss_func = F.nll_loss
    train_loader = DataLoader(datamodule.train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    lr_finder = LRFinder(dummy_model, optimizer, loss_func, device) 
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode='exp')
    best_lr = lr_finder.plot()
    lr_finder.reset()
    return best_lr
    
