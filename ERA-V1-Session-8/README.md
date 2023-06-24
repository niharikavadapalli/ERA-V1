# ERA V1 Session 8

In this session, we build a neural network model using CIFAR10 dataset. We compare results using 3 different types of normalization methods such as Batch Normalization, Layer Normalization and Group Normalization with each of them acheiving more than 70% accuracy under 50000 model parameters and 20 epochs.

# Model Skeleton:

The model we used has the following structure. 

```
C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11
```

Where C is a convolution layer with kernel size of 3 and "c" is a convolutional layer with kernel size 1, P is pooling layer and GAP is Global Average Pooling layer. Te following diagram shows the network architecture and parameters used.

<img width="561" alt="Screenshot 2023-06-23 at 18 47 56" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/14349528-27f5-4bf0-8fb0-bf173c943364">

We trained this model for CIFAR10 dataset to acheive our accuracy goal of 70% using a network of less than 50000 parameters and under 20 epochs.

# Model with Batch Normalization:














