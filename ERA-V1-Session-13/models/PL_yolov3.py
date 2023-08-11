#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:42:02 2023

@author: svaddi
"""
import math
import torch
import torch.nn.functional as F
import utils.config as config
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from models.yolov3 import YOLOv3
from torch.optim.lr_scheduler import OneCycleLR


class LitYolov3(LightningModule):
    def __init__(self, 
                 loss_criterion,
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
        super().__init__()

        self.save_hyperparameters()
        self.loss_criterion = loss_criterion
        self.scaled_anchors = scaled_anchors
        self.threshold = threshold
        self.model = YOLOv3(num_classes=num_classes)
        self.BATCH_SIZE = BATCH_SIZE
        self.best_lr = best_lr
        self.NUM_WORKERS = NUM_WORKERS
        self.EPOCHS = epochs
        self.IMAGE_SIZE = image_size
        self.NUM_CLASSES = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def set_optimizer_and_scheduler(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = (y[0],y[1],y[2])

        out = self.forward(x)
        loss = (
                self.loss_criterion(out[0], y0, self.scaled_anchors[0]) +
                self.loss_criterion(out[1], y1, self.scaled_anchors[1]) +
                self.loss_criterion(out[2], y2, self.scaled_anchors[2])
            )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0
        x, y = batch
        out = self(x)

        for i in range(3):
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i
            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > self.threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        if stage:
            class_acc = (correct_class/(tot_class_preds+1e-16))*100
            no_obj_acc = (correct_noobj/(tot_noobj+1e-16))*100
            obj_acc = (correct_obj/(tot_obj+1e-16))*100
            self.log(f"Class accuracy is: {class_acc:2f}%", prog_bar=True)
            self.log(f"No obj accuracy is: {no_obj_acc:2f}%", prog_bar=True)
            self.log(f"Obj accuracy is: {obj_acc:2f}%", prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        scheduler_dict = {
            "scheduler": self.scheduler,
            "interval": "step",
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
