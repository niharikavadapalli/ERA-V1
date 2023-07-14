# ERA V1 Session 10

In this session, we write our own custrom ResNet architecture for CIFAR10 dataset to acheive a target accuracy of 90% under 24 epochs using One Cycle LR policy.

## Model Skeleton:

The model we used is shown below. 

<img width="708" alt="Screenshot 2023-07-14 at 14 24 32" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/d3a1063c-2ea3-4871-ab72-53b05aa921af">

<img width="565" alt="Screenshot 2023-07-14 at 14 23 56" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/2a1ed7e3-858e-410e-993a-10326427a9f1">

As shown above, the total parameters used are around 6.5M. 


## Image Augmentation:

I have used Albumentations library to augment the train dataset. The below image shows different transforms used for augmentation. The different types of transforms that are used are HorizontalFlip, ShiftScaleRotate and CoarseDropOut.

<img width="1414" alt="Screenshot 2023-06-30 at 12 34 54" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/9bebe736-9796-44a7-bf86-6963bc97d9be">

## Code

```
backpropogation.py - contains methods to train and test, given a model, optimize, test and train loaders.
dataset.py - contains MyDataset class which is a placeholder for dataset
model.py - contains actual model implementation in pytorch
transform.py - contains different transforms used to augment the data. This used Albumentation library for transform functions.
visualize.py - contains all the helper methods to help visualize data and results.

```

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





















