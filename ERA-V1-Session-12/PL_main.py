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
from models.PL_custom_resnet_model import LitResnet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch_lr_finder import LRFinder


def create_pl_model(BATCH_SIZE, NUM_WORKERS, best_lr):
    return LitResnet(lr=0.01, BATCH_SIZE=BATCH_SIZE, best_lr=best_lr)

def train_pl_model(model, datamodule, epochs = 2):
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
        num_sanity_val_steps=0
    )
    
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)
    return trainer

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
    
