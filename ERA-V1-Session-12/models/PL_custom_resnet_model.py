#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:42:02 2023

@author: svaddi
"""

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from models.custom_resnet_model import Net
from torch.optim.lr_scheduler import OneCycleLR


class LitResnet(LightningModule):
    def __init__(self, lr=0.05, BATCH_SIZE = 256, NUM_WORKERS = 1, best_lr = 2.93E-02):
        super().__init__()

        self.save_hyperparameters()
        self.model = Net()
        self.BATCH_SIZE = BATCH_SIZE
        self.best_lr = best_lr
        self.NUM_WORKERS = NUM_WORKERS

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task='multiclass',
                                     num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                self.best_lr,
                pct_start = 5./self.trainer.max_epochs,
                div_factor = 2000,
                three_phase =False,
                final_div_factor = 1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
                anneal_strategy = 'linear',
                verbose=False
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
