import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch

dropout_value = 0.05

def getNorm(output_size, norm='bn', GROUP_SIZE = 1):
  if norm == 'bn':
      n1 = nn.BatchNorm2d(output_size)
  elif norm == 'gn':
      n1 = nn.GroupNorm(GROUP_SIZE, output_size)
  elif norm == 'ln':
      n1 = nn.GroupNorm(GROUP_SIZE, output_size)
  return n1

class Net(nn.Module):
    def __init__(self, norm='bn', GROUP_SIZE = 1):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        # CONVOLUTION 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            getNorm(16, norm, GROUP_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 30, RF = 3

        # CONVOLUTION 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            getNorm(32, norm, GROUP_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 28, RF = 3

        # TRANSITION BLOCK 1
        # CONVOLUTION 4
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 28, RF = 3
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, RF = 6


        # CONVOLUTION BLOCK 2
        # CONVOLUTION 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            getNorm(16, norm, GROUP_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF = 3

        # CONVOLUTION 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            getNorm(32, norm, GROUP_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF = 3

        # CONVOLUTION 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            getNorm(32, norm, GROUP_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF = 3

        # TRANSITION BLOCK 2
        # CONVOLUTION 7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 12, RF = 6
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 12, RF = 6



        # CONVOLUTION BLOCK 3
        # CONVOLUTION 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            getNorm(16, norm, GROUP_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF = 3

        # CONVOLUTION 9
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            getNorm(32, norm, GROUP_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF = 3

        # CONVOLUTION 10
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            getNorm(64, norm, GROUP_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 26, RF = 3


        # OUTPUT BLOCK

        # GAP
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        ) # output_size = 1, RF = 26

        # CONVOLUTION 11
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 2, RF = 26



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.gap(x)
        x = self.convblock11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)