# ERA V1 Session 8

In this session, we build a neural network model using CIFAR10 dataset. We compare results using 3 different types of normalization methods such as Batch Normalization, Layer Normalization and Group Normalization with each of them acheiving more than 70% accuracy under 50000 model parameters and 20 epochs.

## Model Skeleton:

The model we used has the following structure. 

```
C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11
```

Where C is a convolution layer with kernel size of 3 and "c" is a convolutional layer with kernel size 1, P is pooling layer and GAP is Global Average Pooling layer. Te following diagram shows the network architecture and parameters used.

<img width="561" alt="Screenshot 2023-06-23 at 18 47 56" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/14349528-27f5-4bf0-8fb0-bf173c943364">

We trained this model for CIFAR10 dataset to acheive our accuracy goal of 70% using a network of less than 50000 parameters and under 20 epochs.

## Model with Batch Normalization:

With batch normalization after each convolution layer (except for 1x1 conv layers), we were able to acheive a best train accuracy of 77.40% and best test accuracy of 76.96%. As seen above, we used a dropout of 5% after each convolution layer to eliminate any overfitting done by the network. Also performed few data augmentation techniques such as image normalization and rotation to reduce overfitting and make training difficult to network. The images below shows the accuracies and train/test losses of the network.

<img width="906" alt="Screenshot 2023-06-23 at 18 54 41" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/4069bc43-cb8d-49c7-af37-8d4269f7066e">

<img width="1246" alt="Screenshot 2023-06-23 at 18 55 15" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/78f8ba30-2cb1-4660-b101-15e01cbd40fa">

Here is a plot that shows few of the misclassified images.

<img width="663" alt="Screenshot 2023-06-23 at 18 56 24" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/bf27ded5-675b-4767-8066-1922e40fcea5">

Note that the title of each of the figure in the above plot indicate the misclassified image label/actual image label.

## Model with Layer Normalization:

With layer normalization after each convolution layer (except for 1x1 conv layers), we were able to achieve a best train accuracy of 71.50% and best test accuracy of 71.76%. The below image shows the accuracies and train/test losses of the network.

<img width="894" alt="Screenshot 2023-06-23 at 19 00 03" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/017a72cc-bb79-45f7-8fe9-1aeb664ff3c0">

<img width="1270" alt="Screenshot 2023-06-23 at 19 00 18" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/48b9b7c4-f3fa-4c20-82f4-37d6d16ac6fe">

Here is a plot that shows few of the misclassified images.

<img width="658" alt="Screenshot 2023-06-23 at 19 00 33" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/820da115-ebcf-4653-9a61-6bec01d6de90">

Note that the title of each of the figure in the above plot indicate the misclassified image label/actual image label.

## Model with Group Normalization:

With group normalization with group size of 2 after each convolution layer (except for 1x1 conv layers), we were able to achieve a best train accuracy of 71% and best test accuracy of 70.41%. The below image shows the accuracies and train/test losses of the network.

<img width="884" alt="Screenshot 2023-06-23 at 19 04 17" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/eec0fe72-3471-4d7a-ab93-f3d5fa5513df">

<img width="1259" alt="Screenshot 2023-06-23 at 19 04 41" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/bc5c6629-9fd8-4eec-ad37-c52a1ecd86e5">

Here is a plot that shows few of the misclassified images.

<img width="665" alt="Screenshot 2023-06-23 at 19 04 54" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/c3e99031-7b3f-4cd3-a999-42538fd6b506">

Note that the title of each of the figure in the above plot indicate the misclassified image label/actual image label.

## Observation and Comparison:






















