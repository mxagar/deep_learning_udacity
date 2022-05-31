# Pytorch Guide

These are my personal notes taken while following the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

The nanodegree is composed of six modules:

1. Introduction to Deep Learning
2. Neural Networks and Pytorch Guide
3. Convolutonal Neural Networks (CNN)
4. Recurrent Neural Networks (RNN)
5. Generative Adversarial Networks (GAN)
6. Deploying a Model

Each module has a folder with its respective notes. This folder is the one of the **second module** and it contains a Pytorch guide.

Additionally, note that I made many hand-written nortes, which I will scan and push to this repostory.

Here, I refence notebooks that are present in two repositories (both updated, but the second more advanced):

- [DL_PyTorch](https://github.com/mxagar/DL_PyTorch), referenced in the CVND
- [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch) `/intpo-to-pytorch/`, the one used in the DLND

Additionally, in this particular folder, I also collect some examples and summaries made by myself.

## Overview of Contents

1. Introduction
2. Tensors: `Part 1 - Tensors in Pytorch.ipynb`
3. Neural Networks: `Part 2 - Neural Networks in PyTorch.ipynb`
4. Training Neural Networks: `Part 3 - Training Neural Networks.ipynb`
5. Fashion-MNIST Example: `Part 4 - Fashion-MNIST.ipynb`
6. Inference and Validation: `Part 5 - Inference and Validation.ipynb`
7. Saving and Loading Models: `Part 6 - Saving and Loading Models.ipynb`
8. Loading Image Data: `Part 7 - Loading Image Data.ipynb`
9. Transfer Learning: `Part 8 - Transfer Learning.ipynb`

## 1. Introduction

Primarily developed by Facebook AI Research (FAIR).  
Released in 2017.  
Open Source, BSD.  
Very intuitive: similar to Numpy and DL concepts integrate din a more natural way; more intuitive than TensorFlow or Keras.  
Caffe2 was integrated to PyTorch in 2018.  
Main intefarce: Python - it's very Pythonic; C++ interface is available too.  
Main class: Tensors = multidimensional arrays, similar to Numpy's, but they can be operated on CUDA GPUs.  
Automatic differentiation used (autodiff?): derivative used in backpropagation computed in feedforward pass.  

Very interesting Tutorial: [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

Installation:

```bash
conda install pytorch torchvision -c pytorch
```

## 2. Tensors: `Part 1 - Tensors in Pytorch.ipynb`

Tensors are a generalization of arrays or matrices; 1D: column vector, 2D row-col matrix, etc.

```python
%matplotlib inline
import numpy as np
import torch

torch.manual_seed(7)

# Create some tensors
# Torch can take tuples for shapes, numpy doesn't
x = torch.randn((5, 1)) # 5 rows, 1 cols
b = torch.ones((1,1))
w1 = torch.randn_like(x) # rand with same shape as x
w2 = torch.randn_like(x.size()) # rand with same shape as x

# Expected operations are possible, as in numpy
z = torch.sum(w1*x) + b
a = 1/(1+torch.exp(-z)) # sigmoid activation

# Dot and matrix multiplications: matmul
# But size must conincide: change with either
# `view` (just a view), reshape, resize_
# prefer .view() if the new shape is only for the current operation
z = torch.matmul(w1.view(1,5),x) + b

# More operations
x = torch.rand((3, 2))
y = torch.ones(x.size())
z = x + y
# First row
z[0]
# Slicing works as usual
z[:, 1:]
# Methods return new object
z_new = z.add(1) # z + 1, elementwise
# Except when followed by _ == inplace
z.add_(1) # z+1 elementwise, inplace
z.mul_(2) # z*2 elementwise, inplace

# Transform torch <-> numpy
np.set_printoptions(precision=8)
torch.set_printoptions(precision=8)
a = np.random.rand(4,3) # Note: no tuple passed...
b = torch.from_numpy(a)
# BUT: memory is shared between numpy a and pytorch b: 
# changing one affects the other!
```

Example of a simple forward pass in a MLP:

```python
import numpy as np
import torch

torch.manual_seed(7)

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
# Note: here (and in all Udacity ND) the weight matrices are defined
# as the transpose of the Andrew Ng's: W = (input,output)
# Andrew Ng defines them as W = (output,input)
# Probably the Udacity approach is more intuitive
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# Bias terms for hidden and output layers
# Note: Andrew Ng extends the weight matrices to contain the bias
# In Udacity, the bias is simply summed after the matrix multiplication
# and it has always the size (batch_size,output)
# Probably the Udacity approach is more intuitive
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

# Forward pass
z_1 = torch.matmul(features,W1) # (1,3)x(3,2) = (1,2)
z_1 += B1
a_1 = activation(z_1)
z_2 = torch.matmul(a_1,W2) # (1,2)x(2,1) = (1,1)
z_2 += B2
a_2 = activation(z_2) # output: h
```

## 3. Neural Networks: `Part 2 - Neural Networks in PyTorch.ipynb`

Very important notebook.  
A fully connected neural network is built and trained with the MNIST dataset.

The following steps are followed:

1. Download the MNIST Dataset
2. Inspect the images
3. Manual definition of a forward pass
4. Neural Network definition with `torch.nn.Module` in a class
5. Access and modify weights and biases of a network
6. Forward pass
7. Neural Network definition with `torch.nn.Sequential`


```python

import numpy as np
import torch
import helper
import matplotlib.pyplot as plt

### -- 1. Download the MNIST Dataset

# Issue: https://github.com/pytorch/vision/issues/1938
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

from torchvision import datasets, transforms

# Define a transform to normalize the data
# First: convert images to tensors
# Second: normalize them to contain vaues [-1,1]; original data contains [0,1]
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
# batch_size=64 -> trainloader will give us 64 images at a time!
# train=True -> it's going to be used for training!
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

### -- 2. Inspect the images

dataiter = iter(trainloader) # create an interator that yields next batch of images+labels
images, labels = dataiter.next() # get next batch
print(type(images)) # <class 'torch.Tensor'> 
print(images.shape) # torch.Size([64, 1, 28, 28]): batch_size, channels, rows, columns
print(labels.shape) # torch.Size([64])

# Display image with index 1 from batch
# For that, transform it to Numpy and squeeze it (remove single-dimensional entries)
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

### -- 3. Manual definition of a forward pass

# Sigmoid: Map to [0,1]
def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

# Softmax: Multi-class probabilities (multi-class sigmoid)
def softmax(x):
    e = torch.exp(x) # (64,10)
    # Tensor divisions: columns are divided element-wise!
    # Thus: we need to sum across columns (dim=1) and reshape to column size
    return e / torch.sum(e,dim=1).view(x.shape[0],-1) # (64,10) / (64,1) = (64,10)
    # Also:
    # return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

# Flatten images: (64,1,28,28) -> (64,28*28)
# Watch out: resize_ is inplace, but can work
# images.resize_(images.shape[0], images.shape[2]*images.shape[2])
# Anothe option
inputs = images.view(images.shape[0],-1) # -1: infer the rest: 1*28*28 = 784

n_input = 28*28
n_hidden = 256
n_output = 10

W1 = torch.randn((n_input,n_hidden)) # 784x256
W2 = torch.randn((n_hidden,n_output)) # 256x10
B1 = torch.randn((64,n_hidden)) # 64x256
B2 = torch.randn((64,n_output)) # 64x10

z_1 = torch.matmul(inputs,W1) # (64,784) x (784x256) = (64,256)
z_1 += B1
a_1 = sigmoid(z_1)
z_2 = torch.matmul(a_1,W2) # (64,256) x (256,10) = (64,10)
z_2 += B2
a_2 = softmax(z_2)

probabilities = a_2 # (64,10)

### -- 4. Neural Network definition with `torch.nn.Module` in a class

# Here a neural network is defined in a class
# The architecture is:
# fully connected layers with dimensions 28*28 -> 128 -> 64 -> 10

# First layer has as inputs rows x columns of input images: 28x28 = 784
# Last layer must have as outputs number of predicted classes: 10 (0-9)
# Hidden layers defined arbitrarily, but they must match first and last
# Usually, the higher the number of layers and nodes in them, the better
# BUT: most of the times, DL consists in finding the best number of layers, nodes, etc.
# Note: layers have in_features and out_features; instead of unit/neuron layers,
# they are the representation of the weight matrix that connects two layers;
# as such the units are represented by the outputs.
# Activation functions: use ReLU, bacause it's the fastest,
# except in output: softmax / log_softmax (because we want probability of classes)
# Loss function: cross-entropy, if multi-class classification;
# however, better to use log_softmax as last activation and NLLLoss as loss
# (equivalent to cross-entropy)

# Inherit network class from nn.Module
class Network(nn.Module):
    def __init__(self):
        # Call init of upper class: nn.Module
        super().__init__()
        
        # First hidden layer: linear transformation = fully connected
        # 784 -> 128
        # Linear: W(input,output), B(1,output) -> x*W + B
        # W: model.fc1.weight
        # B: model.fc1.bias        
        self.fc1 = nn.Linear(784, 128)
        
        # Second hidden layer
        # 128 -> 64
        self.fc2 = nn.Linear(128, 64)

        # Output layer: units = number of classes
        # 64 -> 10
        self.fc3 = nn.Linear(64, 10)
        
        # We can define activation functions as class objects 
        # but usually, they are used as F functions
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1) # dim=1: sum across columns for softmax
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # Final tensor should have a size batch_size x units: 64 x 10
        # dim=1: sum across columns for softmax
        x = F.softmax(x, dim=1) # alternative: x = self.softmax(x)
                
        return x

# Instantiate network and get architecture summary
model = Network()
model
# Network(
#   (fc1): Linear(in_features=784, out_features=128, bias=True)
#   (fc2): Linear(in_features=128, out_features=64, bias=True)
#   (fc3): Linear(in_features=64, out_features=10, bias=True)
# )

### -- 5. Access and modify weights and biases of a network

print(model.fc1.weight)
print(model.fc1.bias)
# Set biases to all zeros
model.fc1.bias.data.fill_(0)
# Sample from random normal with standard dev = 0.01
model.fc1.weight.data.normal_(std=0.01)

### -- 6. Forward pass

# Grab data 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is:
# (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784)
# to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)

### -- 7. Neural Network definition with `torch.nn.Sequential`

# With `torch.nn.Sequential`, we don't need to define a class
# we simple need to define the forward pass in a sequence

# Hyperparameters for our network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
print(model) # print model summary
print(model.fc1) # print layer named as 'fc1'

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)

```

## 4. Training Neural Networks: `Part 3 - Training Neural Networks.ipynb`

Pytorch has the **autograd** module with which the operations on the tensors are tracked so that the gradient of a tensor `x` can be computed with `x.backward()`.

Thanks to that, we can simply compute the `loss` function after a `forward()` pass and compute its gradient with `loss.backward()`.


```python
# Set the gradient computation of `x` explicitly to false
# at creation
x = torch.zeros(1, requires_grad=False)
# Set the gradient computation of `x` to True
x.requires_grad_(True)
# De-activate gradient in context
with torch.no_grad():
	y = x * 2
# De/Activate gradient computation globally
torch.set_grad_enabled(True)
# Compute and get gradient
z = x ** 2
z.backward()
x.grad # note we get the gradient of x!
```

Then, this gradient can be passed to an optimization function (e.g., gradient descend).

This concepts are applied to implement the training of the network.

Additionally, some improvements are suggested in the last activation and the loss function.

Steps in the notebook:

1. Load data
2. Define network
3. Training
4. Inference

```python
### -- 1. Load data

# Issue: https://github.com/pytorch/vision/issues/1938
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

### -- 2. Define network

# If we wan to use the cross-entropy loss (nn.CrossEntropyLoss),
# the outputs must be the raw outputs (because of how it is defined inside),
# i.e., without activation! These are also called scores.
# Applying nn.CrossEntropyLoss that way is equivalent to using nn.LogSoftmax:
# LogSoftmax = log(softmax())
# It is better to work with LogSoftmax than with probabilities (very small)
# However, note that an output activated with LogSoftmax
# requires the negative log likelihood loss: nn.NLLLoss
# CONCLUSION:
# - use nn.LogSoftmax as last activation: we get logits = model(inputs)
# - use nn.NLLLoss as loss
# - to get probabilities of classes: predictions = torch.exp(logits)

Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)) # the index where the classes are: 1


# Define the loss
criterion = nn.NLLLoss()

### -- 3. Training

# Optimizers require the parameters to optimize and a learning rate
# SGD: Stochastic gradient descend
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Gradients are accumulated pass after pass
        # Reset them always before a pass!
        optimizer.zero_grad()
        
        # Forward pass
        output = model(images)
        
        # Compute loss
        loss = criterion(output, labels)
        
        # Compute gradients
        # Even though we apply it to loss,
        # the gradients of the weights are computed,
        # because they are used to compute the output and the loss
        loss.backward()
        
        # Optimization: update weights
        optimizer.step()
        
        # Accumulated loss for printing after each epoch
        running_loss += loss.item()
    
    # Print the loss after each epoch
    print(f"Epoch {e+1} / {epochs}: Training loss: {running_loss/len(trainloader)}")

### -- 4. Inference

%matplotlib inline
import helper

# Create iterator
dataiter = iter(trainloader)
# Get a batch
images, labels = next(dataiter)

# Flatten first image of the batch
img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities (LogSoftmax),
# need to take exponential for probabilities
ps = torch.exp(logps)
# Visualize with helper module
helper.view_classify(img.view(1, 28, 28), ps)

```

## 5. Fashion-MNIST Example: `Part 4 - Fashion-MNIST.ipynb`



## 6. Inference and Validation: `Part 5 - Inference and Validation.ipynb`



## 7. Saving and Loading Models: `Part 6 - Saving and Loading Models.ipynb`



## 8. Loading Image Data: `Part 7 - Loading Image Data.ipynb`



## 9. Transfer Learning: `Part 8 - Transfer Learning.ipynb`



