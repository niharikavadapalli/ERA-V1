# ERA V1 Session 6

This assignment has two parts. Part 1 contains explanation of how backpropogation works to train a neural network. Part 2 contains explanation of a model that we used to generate an accuracy of > 99.4% on MNIST dataset using less than 20,000 parameters.

# Part 1

The following figure shows a simple neural network with three layers i.e., one input, one hidden and one output layer. Input layer has two inputs i1 and i2, hidden layer has 2 neurons h1 and h2, output layer has two outputs o1, o2 and we have two target values t1 and t2. The network has a total of 8 weights that we train during the backpropogation.

![Screenshot 2023-06-09 at 16 12 01](https://github.com/niharikavadapalli/ERA-V1-Session-5/assets/135390352/c194ec49-f2e1-48c4-8a60-76b73a57c133)

From above network, we can deduce these following values for each neuron. For example, h1 can be calculated as a sum of product of weight w1 with input i1 and product of weight w2 with input i2. a_h1, a_h2, a_o1 and a_o2 are sigmoid activation functions over h1, h2, o1 and o2 respectively. E1 and E2 are calculated losses (squared difference between predicted and actual target value).

![Screenshot 2023-06-09 at 17 20 50](https://github.com/niharikavadapalli/ERA-V1-Session-5/assets/135390352/2c11bc59-8b9f-43ed-97e3-d9be2400642e)

![Screenshot 2023-06-09 at 17 21 15](https://github.com/niharikavadapalli/ERA-V1-Session-5/assets/135390352/e27267c6-f91c-457e-80a8-bdeeb09d3e82)

![Screenshot 2023-06-09 at 17 21 33](https://github.com/niharikavadapalli/ERA-V1-Session-5/assets/135390352/0b9cf5dc-d245-4982-8d27-0f299cd6014c)


 Now we calculate rate of change of total loss E with respect to each of these weights (dE/dW) after each step inorder to minimize the total loss E. Once we have dE/dW for each weight, we update the network simultaneously for all the weights to be used for the next epoch in such a way that the total loss is reduced. Note that we keep all the weights constant during the dE/dW calculation for each weight and update them all simultaneously once it is calculated for all of them.
 Blocks 3 and 6 above shows the dE/dW for each of these weights.
 
 Here is a snapshot of excel sheet in which we have calculated all the backpropogation values for about 70 epochs.
 
 ![Screenshot 2023-06-09 at 17 39 02](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/a436c189-7596-43e9-a04a-829cb000458e)

Now as we increase the learning rate from 0.1 to 10000, we can see how many iterations it takes for the loss to converge close to zero. Also we can observe that for a large value of LR it may not even converge. The figures below shows how loss changes with respect to number of iterations by increasing LR from 0.1 to 10000.

![im1](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/46052c55-c40f-4a68-996d-bb20e22fef90)
![im2](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/2812608d-f226-46c4-bb8e-22d1388a4012)
![im3](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/3509fa7a-bb70-4e49-b2af-fcabdc869a7d)
![im4](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/b7abcc7e-1445-4c0c-8e30-e7f6d1f8a977)
![im5](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/2629609b-3a74-46ee-ba33-712fd759bfb9)
![im6](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/06dff6a6-426b-4fca-8d65-afe94b1b8842)
![im7](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/086eaad6-e62a-4875-8e31-6e99391ef44a)
![im8](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/0e7fe630-0315-4edf-b05b-c768bd15253d)
![im9](https://github.com/niharikavadapalli/ERA-V1/assets/135390352/efb5c517-d5b8-40a4-ae7f-651485f08ea0)

# Part 2
In this section, we train a neural network to predict hand written digits using the MNIST dataset. Our target is to use a network that has less than 20000 parameters and gets a test/validation accurary of > 99.4% under 20 epochs. Inorder to reach such an accuracy, I have tried different architectures as shown below and improved on the network in terms of number of parameters used and accuracy.

## Architecture 1
As part of 1st step, by observing the images in dataset, we can see that we need a receptive field (RF) of close to 7 pixels for the model to detect edges and gradients. So inorder to get a RF of 7, we added 3 convolutional layers followed by a maxpool layer indicating the first block of model that detects edges. Since the image size and complexity is small, we can have just another block of 3 convolutional layers followed by a global average pooling (GAP) followed by a fully connected layer. Each conv layer is followed by a relu and batch normalization except for the conv layer before output.

<img width="563" alt="Screenshot 2023-06-09 at 18 34 01" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/9de00738-64d1-460c-81da-3c7d3ffffb3b">

<img width="563" alt="Screenshot 2023-06-09 at 18 34 30" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/3069a099-2407-418d-b40b-0697e88a70f3">

As we can see from above network with our initial experiment, we need to reduce the number of parameters to reach our target of 20000. To acheive this we reduce the output channel size in our next architecture.

## Architecture 2

Now we try the first block of network with 8 output channels for each convolutional layer and with 16 output channels in our final block which makes the number of parameters to 9770 as shown below.

<img width="505" alt="Screenshot 2023-06-09 at 18 46 09" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/81907b22-7032-4f91-81d3-e8ddaa5e0766">

<img width="555" alt="Screenshot 2023-06-09 at 18 46 23" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/a7d65452-3341-455e-b626-e74812681ce8">

But with above network, we observe that the accuracy doesn't increase after 10 epochs and stays close to 99.36% as shown below.

<img width="856" alt="Screenshot 2023-06-09 at 18 49 40" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/103fb8b0-0c00-4641-9ea6-8af0e34cb387">

## Architecture 3

In this architecture, I improved further by modifying the output channels of all the conv layers to 16 in both blocks. Also removed padding in conv layers of first block so that the output size after each layer reduces which still keeps the number of parameters less than 20000 as shown below.

<img width="597" alt="Screenshot 2023-06-09 at 18 56 34" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/32e21a03-85ac-4665-b7d7-54fdaf3a4f0f">

<img width="597" alt="Screenshot 2023-06-09 at 18 56 47" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/ae8da4f9-b6b7-40da-ba8e-b5a97f4b0cb5">

But with the above changes, I did not observe any major improvement in terms of results as the accuracy stayed around 99% after 10 epochs.

<img width="844" alt="Screenshot 2023-06-09 at 18 58 11" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/5540b8c0-e11c-449b-bb6f-c0600375ddbd">

## Architecture 4

After serveral experiments, drilled down to this final network with 14522 parameters as shown below.

<img width="581" alt="Screenshot 2023-06-09 at 19 01 52" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/d829895a-f105-4dc4-9f33-88e57aff2863">

<img width="581" alt="Screenshot 2023-06-09 at 19 03 12" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/5975e9c4-ec6e-47ac-9411-6dc461ac7fa8">

The above network was able to consistently produce a test accuracy of greater than 99.4% after 10 epochs. 

<img width="852" alt="Screenshot 2023-06-09 at 19 07 50" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/4215a5fe-2860-4dcb-842a-71740d453e40">


## Summary

After trying various experiments with different network configurations, came to a final network that produced an accuracy of more than 99.4% under 20 epochs with 14522 parameters. Also tried with many other architecture modifications with minor changes such as changing number of conv output channels, number of conv layers, padding, maxpool layers, bias and dropouts. Also note that since the training and test accuracy has no significant difference, I didn't had to consider dropout after each conv layer. 






