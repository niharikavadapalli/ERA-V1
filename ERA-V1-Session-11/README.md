# ERA V1 Session 11

In this session, we focus mainly on analysing how GradCam can be used to visually see what the trained model is looking at an image to detect an object in it.

## Model Skeleton:

In this session, we used a ResNet 18 model to run on Cifar10 dataset using oneCycleLR policy to get faster and better accuracy followed by analysis using GradCam. The images below shows the model summary and skeleton.

<img width="753" alt="Screenshot 2023-07-28 at 17 42 15" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/5218c812-adc1-4f86-a48c-a026244c6bbf">

<img width="430" alt="Screenshot 2023-07-28 at 17 43 06" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/d8ae4998-9de7-4d02-9bc0-0486f2fe02b0">

As shown above, the total parameters used are around 11.2M. 


## Image Augmentation:

I have used Albumentations library to augment the train dataset. The below image shows different transforms used for augmentation. The different types of transforms that are used are RandomCrop and CoarseDropOut.

<img width="1324" alt="Screenshot 2023-07-28 at 17 44 33" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/f18f0d54-e3c9-475d-845f-f93ca70578de">


## Code

```
main.py - contains methods to train and test, given a model, optimize, test and train loaders.
utils/dataset.py - contains MyDataset class which is a placeholder for dataset
models/resnet.py - contains actual ResNet model implementation in pytorch
utils/transform.py - contains different transforms used to augment the data. This used Albumentation library for transform functions.
utils/visualize.py - contains all the helper methods to help visualize data and results.
utils/utilities.py - contains helper methods to implement OneCycleLR policy.

```

## Model Training

With the above model architecture and data augmentations with CIFAR10 dataset, I was able to achieve a best train accuracy of 95.15% and a best test accuracy of 92.44% in 20 epochs. Also the model seems to be consistently achieving an accuracy of more than 90% after 18 epochs where it first reached 91.3% as shown in below image. In this model, we used ADAM and CrossEntropyLoss functions to train the data.

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






















