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
