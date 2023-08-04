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

<img width="694" alt="Screenshot 2023-07-28 at 17 45 57" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/cbfa0f72-d03b-4314-bfce-0765aa22743c">

<img width="610" alt="Screenshot 2023-07-28 at 17 46 16" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/e5b70601-d172-46e6-8391-e3791b49c059">

The image below shows the code used for GradCam implementation.

<img width="755" alt="Screenshot 2023-07-28 at 17 47 54" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/65327a63-84ed-48dc-a973-f68112be5f52">


## Observations and Results:

We can see that by using the above model we were able to acheive more than 90% accuracy. The below images shows how the test loss, test accuracy, train loss, train accuracy and learning rates changed across 20 epochs.

<img width="932" alt="Screenshot 2023-07-28 at 17 49 47" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/c8f80d1b-1749-4e78-b384-da19fbc4a1b8">

<img width="458" alt="Screenshot 2023-07-28 at 17 49 57" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/f29f960e-fcd8-4e7e-99d5-da16ddf53df9">


The below image shows some of the misclassified images. Note that the title of each image below indicates predicted image label vs actual image label.

<img width="630" alt="Screenshot 2023-07-28 at 17 50 19" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/630b4534-eeb0-4b0c-aa23-5f999299f3e0">

I used GradCam to see what the model saw in the misclassified images and predicted wrong and the following image shows the heatmap of the area in the image which the model predicted as the object of interest.

<img width="630" alt="Screenshot 2023-07-28 at 17 50 28" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/7e70e0f3-b6e6-43b6-88aa-09107bccd06c">

As we can see from above image, we could improve our image augmentations to eliminate cases such as above. GradCam can be very useful to make such decisions and improve model in terms of correct prediction.























