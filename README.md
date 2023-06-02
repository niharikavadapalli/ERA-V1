# ERA V1 Session 5

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![GitHub license](https://img.shields.io/github/license/TylerYep/torchinfo)](https://github.com/TylerYep/torchinfo/blob/main/LICENSE)


This assignment repo provides information on using a simple neural network created using PyTorch library. In this project, we implement a model that uses MNIST handwritten dataset to predict digits in images. This repo contains three files namely `model.py` , `utils.py` and `S5.ipynb`. The contents and usage of these files are described below.

# Usage

```
File: model.py
```
This file has a neural network class called Net, that contains model with four convolutional layers followed by two fully connected layers. 

```
File utils.py
```
This file has four utility functions that are used to run our model.
-GetCorrectPredCount: This method is used to get the prediction count given the predictions and labels for the data. 
-train: This method is used to train a given model. It takes the training data, device info (CPU, GPU etc.,), optimizer function and loss function as arguments along with the model.
-test: This method is used to infer the test data given a model, test data, device info (CPU, GPU etc.,) and loss function as arguments.
-PlotGraph: This method is used to plot different graphs to analyze the training loss, training accuracy, test loss and test accuracy w.r.t total epochs ran for the model.

```
File: S5.ipynb
```
This file has actual implementation of the neural network. It has all the logic such as creating and loading MNIST dataset, creating test and train data, visualizing the dataset, creating and training model using the model.py and utils.py modules.

# How To Use

```python
from model import Net
import utils

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
# New Line
criterion = F.nll_loss
num_epochs = 20

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  utils.train(model, device, train_loader, optimizer, criterion)
  utils.test(model, device, test_loader, criterion)
  scheduler.step()
```
The above code creates and trains the model for 20 epochs and shows the progress after each epoch in terms of training loss, train accuracy, test loss and test accuracy. We used a learning rate of 0.01 here. Here is a sample output of above step

```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.8416 Batch_id=117 Accuracy=40.03: 100%|██████████| 118/118 [00:22<00:00,  5.22it/s]
Test set: Average loss: 0.6344, Accuracy: 7720/10000 (77.20%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.1780 Batch_id=117 Accuracy=90.04: 100%|██████████| 118/118 [00:23<00:00,  4.98it/s]
Test set: Average loss: 0.1072, Accuracy: 9681/10000 (96.81%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.2334 Batch_id=117 Accuracy=95.66: 100%|██████████| 118/118 [00:23<00:00,  5.11it/s]
Test set: Average loss: 0.0670, Accuracy: 9790/10000 (97.90%)
```

The code below generates the summary of our model.

```python
!pip install torchsummary
from torchsummary import summary

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
            Conv2d-2           [-1, 64, 24, 24]          18,432
            Conv2d-3          [-1, 128, 10, 10]          73,728
            Conv2d-4            [-1, 256, 8, 8]         294,912
            Linear-5                   [-1, 50]         204,800
            Linear-6                   [-1, 10]             500
================================================================
Total params: 592,660
Trainable params: 592,660
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.93
----------------------------------------------------------------
```


# Results

## Model Performance

With our current model predicting on MNIST dataset, here is the model performance observed after running the model for 20 epochs.


```
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/135390352/243007186-abdf0f12-f3b1-4388-9328-a5c9fe7d36a2.png" alt="Alt text">
```
