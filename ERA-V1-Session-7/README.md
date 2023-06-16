# ERA V1 Session 7

In this session, we build up our neural network model incrementally to achieve 99.4% accuracy with less than 8000 parameters under 15 epochs. We use MNIST dataset for all our experiments.

# Model 1

We start with a basic model which includes setting up the dataset, dataloaders and implementing an initial model. We also visualize data and do some analysis to normalize the data.

## Target:

Setting up the initial model and the required steps such as transforms, data loaders, train and test methods, analyse data and run end to end.

## Results:

<img width="573" alt="Screenshot 2023-06-16 at 16 48 03" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/5c26d01a-7bf8-4e44-adf4-c08d85831449">

We used 1.6M parameters as seen above and we were able to get best train accuracy of 99.96% and best test accuracy of 99.31% as shown below.

<img width="947" alt="Screenshot 2023-06-16 at 16 50 48" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/0a35cba8-09ce-4baf-8a51-3e1cf061bbd7">

## Analysis:

Definitely a huge model with lot many parameters and have great scope to shrink it. Also we can see that by the end of 15th epoch the model is clearly overfitting.

# Model 2
