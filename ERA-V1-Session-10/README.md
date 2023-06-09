# ERA V1 Session 10

In this session, we write our own custrom ResNet architecture for CIFAR10 dataset to acheive a target accuracy of 90% under 24 epochs using One Cycle LR policy.

## Model Skeleton:

The model we used is shown below. 

<img width="708" alt="Screenshot 2023-07-14 at 14 24 32" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/d3a1063c-2ea3-4871-ab72-53b05aa921af">

<img width="565" alt="Screenshot 2023-07-14 at 14 23 56" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/2a1ed7e3-858e-410e-993a-10326427a9f1">

As shown above, the total parameters used are around 6.5M. 


## Image Augmentation:

I have used Albumentations library to augment the train dataset. The below image shows different transforms used for augmentation. The different types of transforms that are used are RandomCrop, HorizontalFlip and CoarseDropOut.

<img width="1394" alt="Screenshot 2023-07-14 at 14 27 49" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/39757980-edb5-49ae-b566-6ad48dd955dc">

## Code

```
backpropogation.py - contains methods to train and test, given a model, optimize, test and train loaders.
dataset.py - contains MyDataset class which is a placeholder for dataset
custom_resnet_model.py - contains actual model implementation in pytorch
transform.py - contains different transforms used to augment the data. This used Albumentation library for transform functions.
visualize.py - contains all the helper methods to help visualize data and results.

```

## Model Training

With the above model architecture and data augmentations with CIFAR10 dataset, I was able to achieve a best train accuracy of 96.25% and a best test accuracy of 93.11% in 24 epochs. Also the model seems to be consistently achieving an accuracy of more than 90% after 20 epochs where it first reached 90.1% as shown in below image. In this model, we used ADAM and CrossEntropyLoss functions to train the data.

![Screenshot 2023-07-14 at 14 19 31](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/443de4f9-f16c-44bc-8443-6bc5e6cf3b63)

<img width="733" alt="Screenshot 2023-07-14 at 14 35 54" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/a1b3b746-d9d7-4da7-b74d-54d53992af1b">


The image below shows the parameters used for OneCycleLR policy implementation.

<img width="1387" alt="Screenshot 2023-07-14 at 14 34 07" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/cb5fd6c0-33f5-4688-befb-bd040327a9bd">


## Observations and Results:

We can see that by using One Cycle LR policy, we were able to improve the performance of the network from 87% accuracy in previous model (Session 9) to 93.1% accuracy using our custom ResNet model. The below images show how accuracies and losses changes across epochs.

<img width="1127" alt="Screenshot 2023-07-14 at 14 35 40" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/da467558-f350-4335-a244-220f8da13f20">

The below image shows how the learning rate is changed across 24 epochs.

<img width="533" alt="Screenshot 2023-07-14 at 14 36 37" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/0929d77d-6782-47dd-a7eb-cde41ce6c7d8">

The below image shows some of the misclassified images. Note that the title of each image below indicates predicted image label vs actual image label.

<img width="581" alt="Screenshot 2023-07-14 at 14 37 53" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/38cd7234-17c3-4a81-87bf-ab5407b5d0bc">






















