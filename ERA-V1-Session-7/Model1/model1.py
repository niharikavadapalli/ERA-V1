import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(128, 256, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(256, 512, 3) # 5 > 3 | 32  
        self.conv7 = nn.Conv2d(512, 10, 3) # 3 > 1 | 34 

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)