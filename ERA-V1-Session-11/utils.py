#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:56:18 2023

@author: svaddi
"""

from torch_lr_finder import LRFinder
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

def plot_lr_finder(model, train_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    return optimizer, criterion
    



def get_OneCycleLR(optimizer, EPOCHS, FOUND_LR, train_loader):
    scheduler = OneCycleLR(
            optimizer,
            max_lr=FOUND_LR,
            steps_per_epoch=len(train_loader),
            epochs=EPOCHS,
            pct_start=5/EPOCHS,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )
    return scheduler


