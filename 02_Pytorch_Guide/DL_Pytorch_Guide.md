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
10. Tips and Tricks
11. Lab: Example Projects

## 1. Introduction and Summary

Primarily developed by Facebook AI Research (FAIR).  
Released in 2017.  
Open Source, BSD.  
Very intuitive: similar to Numpy and DL concepts integrate din a more natural way; more intuitive than TensorFlow or Keras.  
Caffe2 was integrated to PyTorch in 2018.  
Main intefarce: Python - it's very Pythonic; C++ interface is available too.  
Main class: Tensors = multidimensional arrays, similar to Numpy's, but they can be operated on CUDA GPUs.  
Automatic differentiation used (autograd?): derivative used in backpropagation computed in feedforward pass.  

Very interesting Tutorial: [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

Installation:

```bash
conda install pytorch torchvision -c pytorch
```

The rest of the sections show how to perform image classification with Pytorch; the typical steps are covered: dataset loading, network architecture definition, training and inference. 

### Summary: `helper.py`, `fc_model.py`

There are two additional files in the repository folder which summarize the complete knowledge of the Udacity lesson on how to use Pytorch for deep learning:

- `fc_model.py`: the definition of a fully connected `Network` class, with a `train()` and `validation()` function. This is the definitive example we should use as blueprint; the content of the file is build step by step in the notebooks `Part 1 - Part 5`. In adition, I copied the functions `save_model()` and `load_checkpoint()` to the module.
- `helper.py`: a helper module mainly with visualization functionalities.

**Those two files and the last two notebooks are a very good summary of how to use Pytorch**:

- `Part 7 - Loading Image Data.ipynb`
- `Part 8 - Transfer Learning.ipynb`

However, they focus only on fully connected / linear networks; CNNs, RNNs, GANs & Co. are covered in dedicated modules.

#### File: `helper.py`:

```python
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable


def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')

def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


```

#### File `fc_model.py`

```python
'''Example of use:

# IMPORTS
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper # visualization utils
import fc_model # model definition, traning, saving, loading

# LOAD DATASET: example, Fashion-MNIST (28x28 pixels, 1 channel, 10 classes)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# CHECK DATSET
image, label = next(iter(trainloader))
print(trainset.classes)
helper.imshow(image[0,:]);

# CREATE NETWORK
input_size = 1*28*28
output_size = 10
hidden_sizes = [512, 256, 128]
model = fc_model.Network(input_size, output_size, hidden_sizes)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAIN (with cross-validation, but without early stopping)
fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

# SAVE
filename = 'my_model_checkpoint.pth'
fc_model.save_model(filname, model, input_size, output_size, hidden_sizes)

# LOAD
model = fc_model.load_checkpoint('checkpoint.pth')
print(model)

# INFER & VISUALIZE
model.eval()
images, labels = next(iter(testloader))
img = images[0]
img = img.view(1, 28*28)
with torch.no_grad():
    output = model.forward(img)
ps = torch.exp(output)
helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')

'''


import torch
from torch import nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)


def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):

    # Check if CUDA GPU available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1
            
            # Transfer to CUDA device if available
            #images, labels = images.to(device), labels.to(device)

            # Flatten images into a channelsxrowsxcols long vector (784 in MNIST 28x28 case)
            pixels = images.size()[1]*images.size()[2]*images.size()[3]
            images.resize_(images.size()[0], pixels)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()

                
def save_model(filepath, model, input_size, output_size, hidden_sizes):
    # Convert model into a dict: architecture params (layer sizes) + state (weight & bias values)
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_sizes,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)

    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # Create a model with given architecture params (layer sizes) + model state (weight & bias values)
    model = Network(checkpoint['input_size'],
                    checkpoint['output_size'],
                    checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model
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
w2 = torch.randn(x.size()) # rand with same shape as x

# Check size/shape
# Resizing and checking the current size
# is very common/necesssary
x.shape
x.size()
# To resize:
# .view() (just a view), .reshape(), .resize_()

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

# Epoch 1 / 5: Training loss: 1.9347652730657094
# Epoch 2 / 5: Training loss: 0.8835022319862837
# Epoch 3 / 5: Training loss: 0.5357787555405326
# Epoch 4 / 5: Training loss: 0.42882247761622677
# Epoch 5 / 5: Training loss: 0.3812579528641091

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

In this notebook, the implementation of the previous one is applied to the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset from Zalando. This dataset is more realistic than then MNIST.

Almost no new things are introduced here, but it is a nice example; yet to be completed with validation.

The following steps are covered (the same as in notebook 3)

1. Load data
2. Define network
3. Training
4. Inference

```python

### -- 1. Load data

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Visualize one image
image, label = next(iter(trainloader))
helper.imshow(image[0,:]);

### -- 2. Define network

# Fast method: Sequential
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Alternative: network class definition
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

model = Classifier()

### -- 3. Training

from torch import optim

# Criterion = Loss
criterion = nn.NLLLoss()

# Optimizers require the parameters to optimize and a learning rate
# Adam: https://pytorch.org/docs/master/generated/torch.optim.Adam.html#torch.optim.Adam
optimizer = optim.Adam(model.parameters(), lr=0.01)
#optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training
epochs = 10
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

# Epoch 1 / 10: Training loss: 0.3848758656650718
# Epoch 2 / 10: Training loss: 0.37357360132531064
# Epoch 3 / 10: Training loss: 0.3789872529981995
# Epoch 4 / 10: Training loss: 0.3728540780574782
# Epoch 5 / 10: Training loss: 0.36859081310631114
# Epoch 6 / 10: Training loss: 0.35351983534057
# Epoch 7 / 10: Training loss: 0.35748081701174217
# Epoch 8 / 10: Training loss: 0.35502494146018776
# Epoch 9 / 10: Training loss: 0.3526441021038017
# Epoch 10 / 10: Training loss: 0.3492871415036828

### -- 4. Inference

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

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
# Visualize with helper module: image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')

```

## 6. Inference and Validation: `Part 5 - Inference and Validation.ipynb`

This notebook introduces two important concepts to the network define in the previous notebook, using the Fashion-MNIST dataset:

- A cross-validation split is tested after every epoch
- Dropout is added after every layer in order to prevent overfitting

The cross-validation test allows to check whether the model is overfitting: when the training split loss decreases while the test split losss increases, we are overfitting.

Note that when dropout is added it needs to be turned off for the cross-validation test or the final inference; that is done with `model.eval() - model.train()`:

- `model.eval()`: evaluation or inference mode, dropout off
- `model.train()`: training mode, dropout on

Another method to prevent overfitting is regularization.

Usual metrics to see how the model performs (to be checked with the test split) are: accuracy, precission, recall, F1, top-5 error rate.

In the following, the notebook is summarized in these steps:

1. Load datasets: train and test split (set correct flag)
2. Model definition with dropout
3. Model training with dropout and cross-validation pass after each epoch
4. Inference (turn-off autograd & dropout) 

```python

### -- 1. Load datasets: train and test split (set correct flag)

import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data: Set train=False
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

### -- 2. Model definition with dropout

from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        # Watch out: deactivate/activate it for cross-validation with
        # model.eval() & model.train
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


### -- 3. Model training with dropout and cross-validation pass after each epoch

model = Classifier()
criterion = nn.NLLLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30

train_losses, test_losses = [], []
for e in range(epochs):
    tot_train_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        tot_train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    else:
        tot_test_loss = 0
        test_correct = 0  # Number of correct predictions on the test set
        # Turn off dropout
        model.eval()
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                tot_test_loss += loss.item()

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_correct += equals.sum().item()
                # Mean computation could also work like this:
                # accuracy = torch.mean(equals.type(torch.FloatTensor))

        # Turn on dropout back
        model.train()
        
        # Get mean loss to enable comparison between train and test sets
        # Filter the loss after each epoch taking the mean, else very noisy!
        train_loss = tot_train_loss / len(trainloader.dataset)
        test_loss = tot_test_loss / len(testloader.dataset)

        # At completion of epoch
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Test Loss: {:.3f}.. ".format(test_loss),
              "Test Accuracy: {:.3f}".format(test_correct / len(testloader.dataset)))

# Plot the evolution of the training & test/cross-validation losses

from matplotlib import pyplot as plt
import numpy as np

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)

### -- 4. Inference (turn-off autograd & dropout)

# Import helper module
import helper

# Turn-off dropout!
model.eval()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Flatten: Convert 2D image to 1D vector
img = img.view(1, 784)

# Calculate the class probabilities (softmax) for img
# Turn off autograd for faster inference
with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)

# Plot the image and probabilities
helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')

```

## 7. Saving and Loading Models: `Part 6 - Saving and Loading Models.ipynb`

This notebook makes use of the model definition module in `fc_model.py`.

In the model class we have been working on, we can distinguish:

- model architecture params: layer sizes: input, output, hidden
- model state: weight and bias values after training

We need to save all of these, because loading weights to a netwrok of another size won't work.

We can play around to see whats' actually inside a model

```python
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())
```

In practice, saving/loading can be summarized to two functions, defined below.

```python

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import helper
import fc_model # fully connected classifier

### -- Load the dataset

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

### -- Create model

input_size = 784
output_size = 10
hidden_sizes = [512, 256, 128]
model = fc_model.Network(input_size, output_size, hidden_sizes)

### -- Train

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

### -- Save & Load

# The model is saved in a dictionary that contains
# (1) the model architecture definition
# (2) the weights and biases

def save_model(filepath, model, input_size, output_size, hidden_sizes):
    # Convert model into a dict: architecture params (layer sizes) + state (weight & bias values)
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_sizes,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)

filepath = 'checkpoint.pth'
save_model(filepath, model, input_size, output_size, hidden_sizes)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

model = load_checkpoint('checkpoint.pth')
print(model)
```

## 8. Loading Image Data: `Part 7 - Loading Image Data.ipynb`

In this notebook a classification example is implemented using the [Dogs-vs-Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset from Kaggle. We needd to create an account at Kaggle and download the dataset.

The notebook shows how to structure a dataset of our own with `torchvision.datasets.ImageFolder`.

First, the kaggle stuff:

```
Create account in kaggle
    mxagar@gmail.com

Kaggle API instructions
    https://github.com/Kaggle/kaggle-api

Short version
    pip install kaggle
    log in to Kaggle > account > create API key -> downloaded kaggle.json
    mv ~/Downloads/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

Go to dataset page: https://www.kaggle.com/c/dogs-vs-cats
Data: Download command; download to 
    kaggle competitions download -c dogs-vs-cats

Unzip dogs-vs-cats.zip and its content.

```

The module `torchvision.datasets.ImageFolder` can be used to load our own datasets; however, the module requires the images to be in class folders:

```
.../root/class-1
    pic1-1.jpg
    pic1-2.jpg
    ...
.../root/class-2
    pic2-1.jpg
    pic2-2.jpg
    ...
.../root/class-3
    ...
```

Unfortunately, the kaggle images are not sorted in class folders and inside the `train/` folder we need to make a train and test split. This could be one with a simple python script. However, Udacity provides an already sorted dataset located in `Cat_Dog_data/`, with the following structure:

```
Cat_Dog_data/
    test/
        cat/
            cat16.jpg
            cat22.jpg
            ...
        dog/
            dog17.jpg
            dog23.jpg
            ...
    train/
        cat/
            cat1.jpg
            cat2.jpg
            ...
        dog/
            dog1.jpg
            dog2.jpg
            ...   
```

Together with the dataset definition we need to pass the `transform` operations to the dataset images: [Pytorch Transforms](https://pytorch.org/vision/stable/transforms.html)

The transforms have several goals:

- Standardize all images: same size, all tensors, etc.
- Normalize the images: the network efficiency improves if the pixel values are in `[-1,1]`
- **Data augmentation**: we can add random rotations or similar, which generalize the network

All in all, everything is summarized in the following:

```python
# Define image folder: inside data_dir, each class should have a subfolder, eg
# path/train/dog, path/train/cat...
data_dir = 'Cat_Dog_data'

# Compose transforms: Select transformations to apply to dataset in a pipeline
# ToTensor: convert into a pytorch tensor
# Normalize: it consists in converting the pixel values to the range [-1,1] to improve
# the network efficiency: input[channel] = (input[channel] - mean[channel]) / std[channel]
# Notes on the normalization:
# - Two tuples are passed: one is the mean, the second the std
# - The elements of the tuple are the channels
# - If the original pixel values are in [0,1], the normalization with mean=0.5 and std=0.5 maps them to [-1,1]
# - Thus, a single channeled image with pixel values [0,1] has: transforms.Normalize((0.5,), (0.5,))
# - Note that the pixel values are often in [0,255]
# - Thus, a single channeled image with pixel values [0,255] would have: transforms.Normalize((0.5*255,), (0.5*255,))
# - When we do transfer learning, we need to use the same normalization as in the trained network!
# - Also, when we do transfer learning the size of the image must match with the size of the input layer!

# Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()]) 
                                       #transforms.Normalize((0.5, 0.5, 0.5), 
                                       #                     (0.5, 0.5, 0.5)])


test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])
                                      #transforms.Normalize((0.5, 0.5, 0.5), 
                                      #                     (0.5, 0.5, 0.5)])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle = True)


# Visualize
data_iter = iter(testloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)
```

## 9. Transfer Learning: `Part 8 - Transfer Learning.ipynb`

This notebook shows how to perform transfer learning with state-of-the-art pre-trained models and how to leverage GPUs for faster trainings.

We can use [Torchvision models](https://pytorch.org/docs/0.3.0/torchvision/models.html) for **transfer learning**. These models are usually trained with [Imagenet](https://image-net.org): 1 million labeled images in 1000 categories; however, they can generalize well to our applications.

For each chosen model, we need to take into account:

- The size of the input image, usuall `224x224`.
- The normalization used in the trained model.
- We need to replace the last layer of the model (the classifier) with our classifier and train it with the images of our application. The weights of the pre-trained network (the backbone) are frozen, not changed; only the weights of th elast classifier we add are optimized.

Available networks:

- AlexNet
- VGG
- ResNet
- SqueezeNet
- Densenet
- Inception v3

This notebook shows how to use the pre-trained [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) model with transfer learning. Basically, we need to: change its final layer with our own classifier that maps the last outputs to our class outputs.

However, note that these pre-trained models are huge; training on CPUs takes forever. Due to that, we can use the workspace GPUs or our own GPUs.

GPU usage: the training can speed up 500x! To that end, we need to transfer all the tensors to the GPU device, or vice versa. The following lines summarize how to use the GPU if we have a CUDA device and we can write a device agnostic model with it.

```python
# Check if we have a CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move images and labels to CUDA GPU - if already there, nothing happens
device = 'cuda'
model.to(device) # the model needs to be transferred once
inputs, labels = inputs.to(device), labels.to(device) # each new batch needs to be transferred

# Move images and labels back to CPU - if already there, nothing happens
device = 'cpu'
model.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```

If we get `RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor`, then we are mixing tensors that are on different devices.

Check the following file to see how to run python/jupyter via SSG on a Jetson Nano (with a CUDA GPU):

`~/Dropbox/Documentation/howtos/jetson_nano_howto.txt`

The following steps are carried out in the notebook:

1. Load the dataset andd define the transforms
2. Load the pre-trained network (DenseNet)
3. Change the classifier of the pre-trained network
4. Train the new classifier with the own dataset
5. Check the accuracy with the test split
6. Inference

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

### -- 1. Load the dataset andd define the transforms

data_dir = 'Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

### -- 2. Load the pre-trained network (DenseNet)

# Load pre-trained model
# Check the sizes of the last classifier layer
model = models.densenet121(pretrained=True)
model

### -- 3. Change the classifier of the pre-trained network

# Freeze parameters of the pre-trained network
# so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Define our own last classifier layers
# Our inputs must match with the ones
# in the pre-trained network (in_features)
# and REPLACE the model.classifier
from collections import OrderedDict
model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 256)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Loss: Classification + LogSoftmax -> NLLLoss
criterion = nn.NLLLoss()

# Optimization: Only train the classifier parameters,
# feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

### -- 4. Train the new classifier with the own dataset

# Training loop params
epochs = 2
steps = 0
running_loss = 0
print_every = 5

# Use GPU if it's available: define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer all tensors to device:
# model, images, labels
model.to(device)

# We can modularize this in a function: train()
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        # NOTE: no resizing done, because the architecture does not require it
        # Always check the input size of the architecture (particularly in transfer learning)
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs) # forward pass
        loss = criterion(logps, labels) # cmopute loss
        
        optimizer.zero_grad() # reset gradients
        loss.backward() # compute gradient / backpropagation
        optimizer.step() # update weights

        running_loss += loss.item()
        
        if steps % print_every == 0:
            # VALIDATION: test cros-validation split
            # We can modularize this in a function: validate()
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()

### -- 5. Check the accuracy with the test split

def check_accuracy_on_test(model, testloader, device):    
    correct = 0
    total = 0
    # Change model to device - cuda if available
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # Change images & labels to device - cuda if available
            # NOTE: no resizing done, because the architecture does not require it
            # Always check the input size of the architecture (particularly in transfer learning)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

check_accuracy_on_test(model, testloader, device)
# Accuracy of the network on the 10000 test images: 97 %

### -- 6. Inference

import numpy as np

model.to(device)
model.eval() # set evaluation/inference mode (no dropout, etc.)

# Get next batch aand transfer it to device
dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

# Calculate the class probabilities (log softmax)
with torch.no_grad():
    output = model(images)

# Get an image from batch
img = images[2]
out = output[2]
# Probabilities
ps = torch.exp(out)

# Plot the image and probabilities
# Note: due to the normalization transform
# the pixel values are not nice to visualize,
# we would need to undo the normalization
ps = ps.data.numpy().squeeze() # convert to numpy
fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
img = img.numpy().squeeze() # convert to numpy
img = np.moveaxis(img, 0, -1) # move axes: (C,W,H) -> (W,H,C)
ax1.imshow(img)
ax1.axis('off')
ax2.barh(np.arange(2), ps)
ax2.set_aspect(0.2)
ax2.set_yticks(np.arange(2))
ax2.set_yticklabels(['cat',
                    'dog'], size='small');
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()

```