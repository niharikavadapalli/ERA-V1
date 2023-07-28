#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 08:56:19 2023

@author: svaddi
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import get_OneCycleLR

class main():

    def __init__(self, model, epochs, batch_size, scheduler, optimizer, criterion, device, train_loader, test_loader, found_lr):
        train_acc = []
        test_acc = []
        train_losses = []
        test_losses = []
        incorrect_examples = []
        incorrect_labels = []
        incorrect_pred = []
        EPOCHS = epochs
        BATCH_SIZE = batch_size
        SCHEDULER = scheduler
        model = model
        optimizer = optimizer
        criterion = criterion
        device = device
        train_loader = train_loader
        test_loader = test_loader
        found_lr = found_lr
    
    def train(self):
      self.model.train()
      
      pbar = tqdm(self.train_loader)
      correct = 0
      processed = 0
    
      for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(self.device), target.to(self.device)
    
        self.optimizer.zero_grad()
    
        y_pred = self.model(data)
    
        loss = self.criterion(y_pred, target)
        self.train_losses.append(loss)
    
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    
        pred = y_pred.argmax(dim = 1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
    
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.4f}')
        self.train_acc.append(100*correct/processed)
        
      return self.train_acc, self.train_losses
    
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                idxs_mask = ((pred == target.view_as(pred))==False).view(-1)
                if idxs_mask.numel(): #if index masks is non-empty append the correspoding data value in incorrect examples
                  self.incorrect_examples.append(data[idxs_mask].squeeze().cpu().numpy())
                  self.incorrect_labels.append(target[idxs_mask].cpu().numpy()) #the corresponding target to the misclassified image
                  self.incorrect_pred.append(pred[idxs_mask].squeeze().cpu().numpy())
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
    
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
    
        self.test_acc.append(100. * correct / len(self.test_loader.dataset))
        return self.test_acc, self.test_losses, self.incorrect_examples, self.incorrect_labels, self.incorrect_pred
    
    def run_model(self):
        for epoch in range(self.EPOCHS):
            print("EPOCH:", epoch)
            self.train()
            self.test()
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            print(f"\n current learing rate is {self.optimizer.param_groups[0]['lr']}")
