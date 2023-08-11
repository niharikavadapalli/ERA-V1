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
from models.PL_yolov3 import LitYolov3
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch_lr_finder import LRFinder


def create_pl_model(loss_criterion,
                    scaled_anchors,
                    threshold,
                    optimizer=None,
                    scheduler=None,
                    BATCH_SIZE = 256, 
                    NUM_WORKERS = 1, 
                    best_lr = 2.93E-02, 
                    epochs = 2, 
                    image_size = 416,
                    num_classes=20):
    return LitYolov3(loss_criterion,
                        scaled_anchors,
                        threshold,
                        optimizer,
                        scheduler,
                        BATCH_SIZE, 
                        NUM_WORKERS, 
                        best_lr, 
                        epochs, 
                        image_size,
                        num_classes)

def train_pl_model(model, datamodule, epochs = 2):
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=2 if torch.cuda.is_available() else None,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
        num_sanity_val_steps=0,
        precision=16
    )
    
    trainer.fit(model, datamodule.train_dataloader, datamodule.val_dataloader)
    trainer.test(model, datamodule.test_dataloader)
    return trainer

def get_max_lr(dummy_model, datamodule,optimizer, criterion, BATCH_SIZE,NUM_WORKERS, device):
    train_loader = datamodule.train_dataloader()
    lr_finder = LRFinder(dummy_model.model, optimizer, criterion, device) 
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode='exp')
    best_lr = lr_finder.plot()
    lr_finder.reset()
    return best_lr
    
