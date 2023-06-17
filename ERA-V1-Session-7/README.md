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

In this model, I try to get the model skeleton right before making further improvements to it. After the analysis from previous step, we can see that we need maybe a Receptive Field(RF) of 5 pixels to detect the edges and gradients and need atleast a RF of 22 to cover the entire image. So I decided to come up with a basic skeleton that covers these without optimizing the model size. I also added Batch Normalization after each conv layer that shown improvement in the overall accuracy and without it the accuracy was just 20%. Also the output block was a combination of (1x1) conv with adaptive average pooling to decrease the channel size and parameters at the end. 

## Target:

Coming up with a basic skeleton keeping the previous analysis in mind to get a RF of atleast 22 by the end of network and having transition layer at RF of 5.

## Results:

<img width="592" alt="Screenshot 2023-06-16 at 17 10 57" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/65b07094-7f2c-4c8b-8fc0-b2b50485643c">

As seen above, the model size is now around 400k parameters. With this, I was able to get the best train accuracy of 99.87% and best test accuracy of 99.57%.

<img width="927" alt="Screenshot 2023-06-16 at 17 13 30" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/ccdaceb5-6f42-4137-92ac-a79f7bad3ab7">

## Analysis:

From the results, it seems that the skeleton worked pretty well getting a better train and test accuracies, although the model is huge compared to our target size and some overfitting. We still need to work on overfitting problem and model size in the coming steps.

# Model 3

In this step, I try to focus on reducing the model parameters to less than 8000.

## Target:

Come up with a lighter model with less than 8k parameters and making sure my model skeleton is intact. 

## Results:

<img width="564" alt="Screenshot 2023-06-16 at 17 26 04" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/b10c7315-af8f-46bf-ae1e-2427027c01be">

As seen above, the model now has 7838 parameters only. I was able to reduce the model size by keeping the input and output channels as low as possible at each conv layer. With this model, I was able to get a best train accuracy of 99.60% and a best test accuracy of 99.22%.

<img width="918" alt="Screenshot 2023-06-16 at 17 34 01" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/f68c3736-a938-48e5-aea0-74473da11344">

## Analysis:

We clearly see that both the train and test accuracies fell down a little due to the decrease in the model size. Still the model did decently well acheiveing 99.6% and 99.22% accuracies for train and test. But we still see there is some overfitting and we need to tackle it as part of our next step. 







