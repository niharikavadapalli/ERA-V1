# ERA V1 Session 12

In this session, we focus mainly on implementing our previous model in Session 10 using Pytorch lightening and we acheived an accuracy of 93% using this model.

## Model Skeleton:

In this session, we used a custom ResNet  model to run on Cifar10 dataset using oneCycleLR policy to get faster and better accuracy followed by analysis using GradCam in pytorch lightening. The images below shows the model summary and skeleton.

<img width="1125" alt="Screenshot 2023-08-07 at 09 39 13" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/3ee2f538-8262-4140-995b-e08bf2b14e2d">

As shown above, the total parameters used are around 6.5M. 

## Code

```
PL_main.py - contains methods to create model, datamodule and train the model.
models/PL_custom_resnet_model.py - contains custom resnet model in Pytorch lightening.
models/custom_resnet_model.py - contains actual ResNet model implementation in pytorch
utils/datamodule.py - contains creation of custom data module for cifar10 dataset
utils/visualize.py - contains all the helper methods to help visualize data and results.

```

## Model Training

With the above model architecture and data augmentations with CIFAR10 dataset, I was able to achieve a best train accuracy of 96.15% and a best test accuracy of 93.11% in 24 epochs. Also the model seems to be consistently achieving an accuracy of more than 90% after 20 epochs where it first reached 91.3% as shown in below image. In this model, we used ADAM and CrossEntropyLoss functions to train the data.

<img width="736" alt="Screenshot 2023-08-07 at 09 37 15" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/18e6708e-2bef-41a5-b145-014254cae24c">

<img width="1483" alt="Screenshot 2023-08-07 at 09 47 34" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/61f7e6b9-6e23-4dc4-98be-e0770e3489c6">



## Observations and Results:

We can see that by using the above model we were able to acheive more than 90% accuracy. The below images shows how the test loss, test accuracy, train loss, train accuracy and learning rates changed across 20 epochs.

<img width="706" alt="Screenshot 2023-08-07 at 09 50 20" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/07db1ce9-d71b-4636-b93d-91da9e332aa2">

<img width="706" alt="Screenshot 2023-08-07 at 09 50 14" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/e269d70b-4e01-46fd-89f6-80a60160814f">


The below image shows some of the misclassified images. Note that the title of each image below indicates predicted image label vs actual image label.

<img width="686" alt="Screenshot 2023-08-07 at 09 49 40" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/2adea32c-3516-42d5-8723-2f4378cd6db0">


I used GradCam to see what the model saw in the misclassified images and predicted wrong and the following image shows the heatmap of the area in the image which the model predicted as the object of interest.

<img width="706" alt="Screenshot 2023-08-07 at 09 49 50" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/426302a6-1af1-4a7c-b237-abceadf177db">


As we can see from above image, we could improve our image augmentations to eliminate cases such as above. GradCam can be very useful to make such decisions and improve model in terms of correct prediction.























