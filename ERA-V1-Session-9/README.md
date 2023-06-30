# ERA V1 Session 9

In this session, we try to improve our previous model on CIFAR10 dataset to achieve an accuracy of more than 85% under 200k parameters and by using dilated and deptwise separable convolutions.

## Model Skeleton:

The model we used has the following structure. 

```
C1 C2 C3 C4 GAP c5
```

Where each C is a convolution block with 3 convolution layers. The first convolution layer in each block C (except C1) is a combination of a conv2D layer and conv2D layer with dilation 2. The second convolution layer in each block C (except C1) block is a combination of a conv2D layer with depthwise separable convolution and a conv2D layer with dilation 4. The third convolution layer in each block C (except C1) is a combination of a conv2D layer and conv2D layer with dilation 8. The C1 block has one conv2D layer added to a conv2D layer of dilation 2 and dilation 4. The final C4 block is followed by a Global Average Pooling (GAP) layer and a 1x1 convolution to number of classes. The following diagram shows the parameters used and network structure.

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]             288
            Conv2d-6           [-1, 32, 32, 32]           1,024
       BatchNorm2d-7           [-1, 32, 32, 32]              64
              ReLU-8           [-1, 32, 32, 32]               0
           Dropout-9           [-1, 32, 32, 32]               0
           Conv2d-10           [-1, 32, 32, 32]           9,216
      BatchNorm2d-11           [-1, 32, 32, 32]              64
             ReLU-12           [-1, 32, 32, 32]               0
          Dropout-13           [-1, 32, 32, 32]               0
           Conv2d-14           [-1, 32, 32, 32]           9,216
      BatchNorm2d-15           [-1, 32, 32, 32]              64
             ReLU-16           [-1, 32, 32, 32]               0
          Dropout-17           [-1, 32, 32, 32]               0
           Conv2d-18           [-1, 32, 32, 32]           9,216
      BatchNorm2d-19           [-1, 32, 32, 32]              64
             ReLU-20           [-1, 32, 32, 32]               0
          Dropout-21           [-1, 32, 32, 32]               0
           Conv2d-22           [-1, 32, 30, 30]           9,216
      BatchNorm2d-23           [-1, 32, 30, 30]              64
             ReLU-24           [-1, 32, 30, 30]               0
          Dropout-25           [-1, 32, 30, 30]               0
           Conv2d-26           [-1, 32, 30, 30]           9,216
      BatchNorm2d-27           [-1, 32, 30, 30]              64
             ReLU-28           [-1, 32, 30, 30]               0
          Dropout-29           [-1, 32, 30, 30]               0
           Conv2d-30           [-1, 32, 30, 30]             288
           Conv2d-31           [-1, 32, 30, 30]           1,024
      BatchNorm2d-32           [-1, 32, 30, 30]              64
             ReLU-33           [-1, 32, 30, 30]               0
          Dropout-34           [-1, 32, 30, 30]               0
           Conv2d-35           [-1, 32, 30, 30]           9,216
      BatchNorm2d-36           [-1, 32, 30, 30]              64
             ReLU-37           [-1, 32, 30, 30]               0
          Dropout-38           [-1, 32, 30, 30]               0
           Conv2d-39           [-1, 32, 30, 30]           9,216
      BatchNorm2d-40           [-1, 32, 30, 30]              64
             ReLU-41           [-1, 32, 30, 30]               0
          Dropout-42           [-1, 32, 30, 30]               0
           Conv2d-43           [-1, 32, 30, 30]           9,216
      BatchNorm2d-44           [-1, 32, 30, 30]              64
             ReLU-45           [-1, 32, 30, 30]               0
          Dropout-46           [-1, 32, 30, 30]               0
           Conv2d-47           [-1, 32, 28, 28]           9,216
      BatchNorm2d-48           [-1, 32, 28, 28]              64
             ReLU-49           [-1, 32, 28, 28]               0
          Dropout-50           [-1, 32, 28, 28]               0
           Conv2d-51           [-1, 32, 28, 28]           9,216
      BatchNorm2d-52           [-1, 32, 28, 28]              64
             ReLU-53           [-1, 32, 28, 28]               0
          Dropout-54           [-1, 32, 28, 28]               0
           Conv2d-55           [-1, 32, 28, 28]             288
           Conv2d-56           [-1, 32, 28, 28]           1,024
      BatchNorm2d-57           [-1, 32, 28, 28]              64
             ReLU-58           [-1, 32, 28, 28]               0
          Dropout-59           [-1, 32, 28, 28]               0
           Conv2d-60           [-1, 32, 28, 28]           9,216
      BatchNorm2d-61           [-1, 32, 28, 28]              64
             ReLU-62           [-1, 32, 28, 28]               0
          Dropout-63           [-1, 32, 28, 28]               0
           Conv2d-64           [-1, 32, 28, 28]           9,216
      BatchNorm2d-65           [-1, 32, 28, 28]              64
             ReLU-66           [-1, 32, 28, 28]               0
          Dropout-67           [-1, 32, 28, 28]               0
           Conv2d-68           [-1, 32, 28, 28]           9,216
      BatchNorm2d-69           [-1, 32, 28, 28]              64
             ReLU-70           [-1, 32, 28, 28]               0
          Dropout-71           [-1, 32, 28, 28]               0
           Conv2d-72           [-1, 32, 26, 26]           9,216
      BatchNorm2d-73           [-1, 32, 26, 26]              64
             ReLU-74           [-1, 32, 26, 26]               0
          Dropout-75           [-1, 32, 26, 26]               0
           Conv2d-76           [-1, 32, 26, 26]           9,216
      BatchNorm2d-77           [-1, 32, 26, 26]              64
             ReLU-78           [-1, 32, 26, 26]               0
          Dropout-79           [-1, 32, 26, 26]               0
           Conv2d-80           [-1, 32, 26, 26]             288
           Conv2d-81           [-1, 32, 26, 26]           1,024
      BatchNorm2d-82           [-1, 32, 26, 26]              64
             ReLU-83           [-1, 32, 26, 26]               0
          Dropout-84           [-1, 32, 26, 26]               0
           Conv2d-85           [-1, 32, 26, 26]           9,216
      BatchNorm2d-86           [-1, 32, 26, 26]              64
             ReLU-87           [-1, 32, 26, 26]               0
          Dropout-88           [-1, 32, 26, 26]               0
           Conv2d-89           [-1, 64, 26, 26]          18,432
      BatchNorm2d-90           [-1, 64, 26, 26]             128
             ReLU-91           [-1, 64, 26, 26]               0
          Dropout-92           [-1, 64, 26, 26]               0
           Conv2d-93           [-1, 64, 26, 26]          18,432
      BatchNorm2d-94           [-1, 64, 26, 26]             128
             ReLU-95           [-1, 64, 26, 26]               0
          Dropout-96           [-1, 64, 26, 26]               0
AdaptiveAvgPool2d-97             [-1, 64, 1, 1]               0
           Conv2d-98             [-1, 10, 1, 1]             640
================================================================
Total params: 192,672
Trainable params: 192,672
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 20.98
Params size (MB): 0.73
Estimated Total Size (MB): 21.72
----------------------------------------------------------------

As shown above, the total parameters used are 192k.

## Model training:

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

We can see that the performance of our network with Batch Normalization was far more better than the networks with layer and group normalizations achieving a best accuracy of 77%. This validates that the batch normalization is best suitable for the convolutional networks since it is applied accross batches of images rather than applying accross each image and taking mean for all images. And intuitively it seems that the layer and group normalization would work much better for language and contextual machine learning models.





















