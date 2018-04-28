# Rudimentary library for neural networks, implemented in Numpy

The generic neural network is implemented in class format, so it can be easily plugged into any project. 

The purpose of the library is twofold: i) to have a tool that runs without the need to install extra packages (which in some network may be restricted by the administrator) and ii) to familiarise myself with the mechanics of implementing and training neural networks.    

It includes elements such as:
- **layers**: fully-connected, convolutional;
- **optimisers**: SGD, Adam; 
- **regularization**: L2, dropout;
- **activation functions**: relu, leaky relu, tanh, sigmoid, softmax;
- **cost functions**: mean squared errors, cross-entropy.    

Feedback is welcome.  
[@flankme](https://github.com/flankme)

## Quick start
Here is a quick way to test the library:
1) Download all files from the repository;
2) In the same folder, save a copy of MNIST's dataset in csv format under the name `MNISTtrain.csv`. It can be downloaded from [LeCun's website](http://yann.lecun.com/exdb/mnist/);
3) Finally, launch the script `QUICK_START.py`. 

## Requirements
- Python
- Numpy
