# ERA V1 Session 13

In this session, I implement Yolo V3 object detection model in pytorch lightening and run on PASCAL VOC dataset and run various experiments around data augmentation techniques like mosaic to achieve a better accuracy.

## Model Skeleton:

In this session, we used a Yolo V3 model to run on PASCAL VOC dataset using oneCycleLR and mosaic data augmentation technique. The images below shows the model summary of implemented Yolo V3 model.

<img width="1539" alt="Screenshot 2023-08-18 at 18 32 39" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/a60cc271-9573-44c3-b845-0483718454c5">


As shown above, the total parameters used are around 61.6M. 

## Code

```

PL_main.py - contains methods to create model, datamodule and train the model.
models/PL_yolov3.py - contains yolo model in Pytorch lightening.
models/yolov3.py - contains actual yolo model implementation in pytorch
utils/datamodule.py - contains creation of custom data module for pascal voc dataset
utils/visualize.py - contains all the helper methods to help visualize data and results.
utils/utils.py - contains all the helper methods such as plotting the predictions, calculating iou etc.
S13_mosaic.ipynb and S13_normal.ipynb - contains the results of the model using a normal vs mosaic data augmentation on pascal voc dataset.

```

## Model Training

With the above model architecture and data augmentations with PASCAL VOC dataset, here is the image showing accuracies of model when run for 40 epochs.

<img width="496" alt="Screenshot 2023-08-18 at 18 39 06" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/dc61ba10-acdb-4e08-b668-dd58b8d632ae">

On test dataset, the model was able to achieve Class accuracy of 83.8%, No Obj accuracy of 98% and Obj accuracy of 79.8% respectively with a MAP score of 0.39.


## Observations and Results:

The following images shows the model performance and visualizations observed during the model run. We can find more detailed results in the ipynb files of this session.

<img width="547" alt="Screenshot 2023-08-18 at 18 42 57" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/5d97d58a-de7d-4e72-bdea-0836e48417f0">

<img width="624" alt="Screenshot 2023-08-18 at 18 43 08" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/742b5926-df5d-4143-87b9-fa2c82395203">

























