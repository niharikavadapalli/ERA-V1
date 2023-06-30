# ERA V1 Session 9

In this session, we try to improve our previous model on CIFAR10 dataset to achieve an accuracy of more than 85% under 200k parameters and by using dilated and deptwise separable convolutions.

## Model Skeleton:

The model we used has the following structure. 

```
C1 C2 C3 C4 GAP c5
```

Where each C is a convolution block with 3 convolution layers. The first convolution layer in each block C (except C1) is a combination of a conv2D layer and conv2D layer with dilation 2. The second convolution layer in each block C (except C1) block is a combination of a conv2D layer with depthwise separable convolution and a conv2D layer with dilation 4. The third convolution layer in each block C (except C1) is a combination of a conv2D layer and conv2D layer with dilation 8. The C1 block has one conv2D layer added to a conv2D layer of dilation 2 and dilation 4. The final C4 block is followed by a Global Average Pooling (GAP) layer and a 1x1 convolution to number of classes. The following diagram shows the parameters used and network structure.

![Screenshot 2023-06-30 at 12 27 37](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/06300dec-e0d0-450c-b258-82f5c96a7d98)

As shown above, the total parameters used are 192k.

## Image Augmentation:

I have used Albumentations library to augment the train dataset. The below image shows different transforms used for augmentation. The different types of transforms that are used are HorizontalFlip, ShiftScaleRotate and CoarseDropOut.

<img width="1414" alt="Screenshot 2023-06-30 at 12 34 54" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/9bebe736-9796-44a7-bf86-6963bc97d9be">


## Model Training

With the above model architecture and data augmentations with CIFAR10 dataset, I was able to achieve a best train accuracy of 84.66% and a best test accuracy of 87.31% at around 47 epochs. Also the model seems to be consistently achieving an accuracy of more than 85% after 28 epochs where it first reached 86.51% (tested till 50 epochs run) as shown in below images.

![Screenshot 2023-06-30 at 14 41 08](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/8fa79cc1-2903-415b-91b8-9d80589fe2a6)

![Screenshot 2023-06-30 at 14 44 48](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/08dcef10-4608-4580-a21f-e5c0d214d5b5)

![Screenshot 2023-06-30 at 14 44 12](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/7cb2fca7-bb93-4134-9814-67712d175fa8)


## Observations and Results:

We can see that by adding dilated convolutions and depthwise separable convolutions, we were able to improve the performance of the network from 75% accuracy in previous model (Session 8) to 87.3% accuracy. We were also able to achieve 85% accuracy within first 25 epochs and with network under 200k parameters. The below images show how accuracies and losses changes across epochs.

![Screenshot 2023-06-30 at 14 50 55](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/b482fb66-bde7-4b07-a11e-e516ec03a2a9)

The below image shows some of the misclassified images. Note that the title of each image below indicates predicted image label vs actual image label.

![Screenshot 2023-06-30 at 14 52 40](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/75b576ac-33dd-4c0c-b6d7-6d177557e8c9)





















