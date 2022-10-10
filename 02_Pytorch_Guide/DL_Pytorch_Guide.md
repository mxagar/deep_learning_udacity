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

Additionally, note that I made many hand-written notes, which I will scan and push to this repository.

Here, I reference notebooks that are present in two repositories (both updated, but the second more advanced):

- [DL_PyTorch](https://github.com/mxagar/DL_PyTorch), referenced in the CVND
- [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch) `/intpo-to-pytorch/`, the one used in the DLND

## Summary and More Applications

The following files in the current folder give a very nice overview/summary of how Pytorch is used for image classification:

- `fc_model.py`: complete pipeline of a fully connected notebook for image classification
- `helper.py`: visualization.
- `Part 7 - Loading Image Data.ipynb`: dealing with custom datasets.
- `Part 8 - Transfer Learning.ipynb`: transfer learning example with a CNN backbone for image classification.

However, many applications go beyond those use cases. To that end, I will collect in the folder `./lab` more blueprints/examples of different applications.

Please, go to the `./lab` folder are read the `README.md` there to get more information.

## Overview of Contents

1. Introduction and Summary
    - File: `helper.py`
    - File `fc_model.py`
2. Tensors: `Part 1 - Tensors in Pytorch.ipynb`
3. Neural Networks: `Part 2 - Neural Networks in PyTorch.ipynb`
4. Training Neural Networks: `Part 3 - Training Neural Networks.ipynb`
5. Fashion-MNIST Example: `Part 4 - Fashion-MNIST.ipynb`
6. Inference and Validation: `Part 5 - Inference and Validation.ipynb`
7. Saving and Loading Models: `Part 6 - Saving and Loading Models.ipynb`
8. Loading Image Data: `Part 7 - Loading Image Data.ipynb`
9. Transfer Learning: `Part 8 - Transfer Learning.ipynb`
    - Notes on Fine Tuning
10. Convolutional Neural Networks (CNN)
    - `Conv2d`
    - `MaxPool2d`
    - Linear Layer and Flattening
    - Example of a Simple Architecture
    - Summary of Guidelines for the Architecture Definitions
    - Guidelines on Training and Hyperparameter Selection
11. Weight Initialization
12. Using the Jetson Nano (CUDA)
13. Recursive Neural Networks (RNN)
    - Introduction: Simple RNNs and LSTMs
    - Defining an LSTM cell in Pytorch
    - Examples
        - Code / Notebooks
14. Recommendations for Hyperparameter Tuning
15. Vanilla Inference Pipeline and Artifact
16. Cloud Computing with AWS
17. Beyond Classification: Object Detection and Semantic Segmentation

Appendices:

- Tips and Tricks
    - Number of Model Parameters
    - Running average loss
- Lab: Example Projects
- Important Links

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

However, they focus only on fully connected / linear networks; CNNs, RNNs, GANs & Co. are covered in dedicated modules and with example projects in Section 10: "Lab: Example Projects".

#### File: `helper.py`

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
    #model.to(device, dtype=torch.float)
    
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1
            
            # Transfer to CUDA device if available
            #images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)

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
    # If we saved the model in a CUDA device, we need to map it to CPU
    # checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
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
# If the matrices are batched, multiplication is done for each batch
# >>> tensor1 = torch.randn(10, 3, 4)
# >>> tensor2 = torch.randn(10, 4, 5)
# >>> torch.matmul(tensor1, tensor2).size()
# torch.Size([10, 3, 5])

# Transpose: x.t(), or:
# >>> x = torch.randn(2, 3)
# >>> x
# tensor([[ 1.0028, -0.9893,  0.5809],
#         [-0.1669,  0.7299,  0.4942]])
# >>> torch.transpose(x, 0, 1) # dimension 0 and 1 swapped
# tensor([[ 1.0028, -0.1669],
#         [-0.9893,  0.7299],
#         [ 0.5809,  0.4942]])

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
# NOTE: I imporved the accuracy from 72%->93% with two changes:
# - add nn.Dropout(0.2) after fc2
# - increase hidden nodes: 784->512->256->10
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

# Set in eval mode! (turn off dropout, etc.)
model.eval()

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

# Set in eval mode! (turn off dropout, etc.)
model.eval()

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

### 6.1 Three Splits: Train, Validation, Test

In reality, we should split out dataset in 3 exclusive groups:

1. Training split: to train.
2. Validation split: to test how well the model generalizes and to choose between hyperparameters.
3. Test split: to evaluate the final model performance.

The training is performed with the training split, while we continuously (e.g., after each epoch) check the validation loss of the model so far. If the model starts overfitting, the training loss will decrease while the validation loss will start increasing. The idea is to save the weights that yield the smallest validation loss. We can do it with early stopping or just saving the weights of the best epoch.

Additionally, we can test different hyperparameters and architectures; in that case, we choose the architecture and set hyperparameters that yield the lowest validation loss.

As we see, the final choice is influenced by teh validation split; thus, the model is balanced in favor of the validation split. That is why we need the last split, the test split: the real performance of our model needs to be validated by a dataset which has never been seen.

I understand that the 3 splits start making sense when we try different hyperparameters and architectures; otherwise, 2 splits are quite common.

Usually, the validation split is taken from the train split; especially, if we have already train and test splits. To that end, the `SubsetRandomSampler` can be used.

```python
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)
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
    # If we saved the model in a CUDA device, we need to map it to CPU
    # checkpoint = torch.load(filepath, map_location=torch.device('cpu'))    
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

Another very nice application of the [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) is in the [CVND Face keypopint detection project](https://github.com/mxagar/P1_Facial_Keypoints).

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
# - Note that transforms.ToTensor() maps the pixel values to [0,1]!
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

# Access class names
train_data.classes

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
model.to(device, dtype=torch.float) # the model needs to be transferred once
inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float) # each new batch needs to be transferred

# Move images and labels back to CPU - if already there, nothing happens
device = 'cpu'
model.to(device, dtype=torch.float)
inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
```

If we get `RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor`, then we are mixing tensors that are on different devices.

See Section 12 or check the following file to see how to run python/jupyter via SSH on a Jetson Nano (with a CUDA GPU):

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
# Note that we can access the layers and layer groups: model.classifier ...
model = models.densenet121(pretrained=True)
print(model)
# It is important to print the model and to take the name of the last layer
# In the case of DenseNet, that's 'classifier'; for ResNet50 it's 'fc'
# We need to replace that last layer with a Sequential that maps the nodes to our desired outputs

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

# This should be superfluous, but just in case
for param in model.classifier.parameters():
    param.requires_grad = True

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
model.to(device, dtype=torch.float)

# We can modularize this in a function: train()
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        # NOTE: no resizing done, because the architecture does not require it
        # Always check the input size of the architecture (particularly in transfer learning)
        inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
        
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
                    inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
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
    model.to(device, dtype=torch.float)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # Change images & labels to device - cuda if available
            # NOTE: no resizing done, because the architecture does not require it
            # Always check the input size of the architecture (particularly in transfer learning)
            images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

check_accuracy_on_test(model, testloader, device)
# Accuracy of the network on the 10000 test images: 97 %

### -- 6. Inference

import numpy as np

model.to(device, dtype=torch.float)
model.eval() # set evaluation/inference mode (no dropout, etc.)

# Get next batch aand transfer it to device
dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)

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

### Notes on Fine Tuning

Transfer learning with a frozed backbone pre-trained network can generalize to our applications well if

- we have few classes (size)
- and the features of our classes are similar to those in the images of ImageNet (similarity).

In general, depending on how the **size** and **similarity** factors are, we should follow different approaches, sketched in this matrix:

![Transfer learning approach](./pics/transfer_learning_matrix.png)

Note that Small datasets (approx. 2k images) risk overfitting; the solution consists in freezing the pre-trained weights, no matter at which depth we cut the backbone network.

Summary of approaches:

1. Small dataset, similar features: Transfer learning with complete backbone
    - Remove last classification linear layers
    - Freeze pre-trained weights
    - Add new linear layers with random weights for classification, which end up in the desired class nodes
    - Train new classification layers
2. Small dataset, different features: Transfer learning with initial part of the backbone
    - Slice off near the begining of the pre-trained network
    - Freeze pre-trained weights
    - Add new linear layers with random weights for classification, which end up in the desired class nodes; do not add convolutional layers, instead use the low level features!
    - Train new classification layers
3. Large dataset, similar features: Fine tune
    - Like case 1, but we don't freeze the backbone weights, but start training from their pre-trained state.
4. Large dataset, different features: Fine tune or Re-train
    - Remove the last fully connected layer and replace with a layer matching the number of classes in the new data set.
    - Re-train the network from the scratch with random weights.
    - Or: perform as in case 3.

## 10. Convolutional Neural Networks (CNNs)

The two most important layers in CNNs are [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) and [MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html). When we instantiate them, we need to take into account their default parameters!

```python
import torch.nn as nn

# Sizes, padding, stride can be tuples!
# Look at default values!
nn.Conv2d(in_channels=1,
          out_channels=32,
          kernel_size=5)
          #stride=1,
          #padding=0,
          #dilation=1,
          #bias=True,
          #padding_mode='zeros',
          #...

# Sizes, padding, stride can be tuples!
# Look at default values!
nn.MaxPool2d(kernel_size=2,
             stride=2)
             #passing=0,
             #...
```

### `Conv2d`

The **channels** refer to the depth of the filter. Notes:

- A grayscale image will have `in_channel = 1` in the first `Conv2d`.
- A color image will have `in_channel = 3` in the first `Conv2d`.
- **The filter depths usually increase in sequence like this: 16 -> 32 -> 64**.

**Padding** is a very important argument in `Conv2d`, because with it we can control the size (WxH) of the output feature maps. Padding consists in adding a border of pixels around an image. In PyTorch, you specify the size of this border.

The simplified formula for the output size is (`dilation=1`):

`W_out = (W_in + 2P - F)/S + 1`

- `W_in`:
- `P`: padding width, **default is 0**
- `F`: kernel/filter size, usually odd numbers; twice 3 is better than once 5, because less parameters and same result!
- `S`: stride, usually left in the **default 1**

Usually, **preserving the sizes leads to better results**! That way, we don't loose information that we would have lost without padding. Thus, we need to:

- Use an odd kernel size: 3, 5, etc.; better 3 than 5.
- Define the padding as the border around the anchor pixel of the kernel: 3->1, 5->2

Since the default padding method is `'zeros'`, the added border is just zeros.

Additionally, after a `Conv2d`, a `relu()` activation is applied!

Which is the number of parameters in a convolutional layer?

`W_out*F*F*W_in + W_out`

- The last term `W_out` is active when we have biases: we have a bias for each output feature map.
- The first term is the pixel area of a filter x `in_channels` x `out_channels`; basically, we apply a convolution `out_channels` times.

### `MaxPool2d`

Usually a `MaxPool2d` that halvens the size is chosen, i.e.:

- `kernel_size = 2`
- `stride = 2`

A `MaxPool2d` can be defined once and used several times; it comes after the `relu(Conv2d(x))`.


### Linear Layer and Flattening

After the convolutional layers, the 3D feature maps need to be reshaped to a 1D feature vector to enter into a linear layer. If padding has been applied so that the size of the feature maps is preserved after each `Conv2d`, we only need to compute the final size taking into account the effects of the applied `MaxPool2d` reductions and the final depth; otherwise, the convolution resizing formmula needs to be applied carefully step by step.

Once we have the size, we compute the number of pixels in the final set of feature maps:

`N = W x H x D`

The linear layer after the last convolutional layer is defined as follows:

```python
nn.Linear(N, linear_out)
# linear_out is the number of nodes we want after the linear layer
# it could be the number of classes
# if this is the last linear layer
```

In the `forward()` function, the flattening is simpler, because we can query the size of the vector:

```python
x = x.view(x.size(0), -1)
# x.size(0): batch size
# -1: deduce how many pixels after dividing all items by the batch size, ie.: W x H x D
```

### Example of a Simple Architecture

```python
import torch.nn as nn
import torch.nn.functional as F

# One convolutional layer applied an 12x12 image for a regression
# - input_size = 12
# - grayscale image: in_channles = 1
# - regression of variable n_classes

# Note: x has this shape: batch_size x n_channels x width x height
# Usually, the batch_size is ignored in the comments, but it is x.size(0)!

class Net(nn.Module):

    def __init__(self, n_classes):
        super(Net, self).__init__()

        # 1 input image channel (grayscale)
        # 32 output channels/feature maps
        # 5x5 square convolution kernel
        # W_out = (W_in + 2P - F)/S + 1
        # W_out: (input_size + 2*0 - 5)/1 + 1 = 8
        # WATCH OUT: default padding = 0!
        self.conv1 = nn.Conv2d(1, 32, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        # output size: 4
        # Note: there is no relu after MaxPool2d!
        self.pool = nn.MaxPool2d(2, 2)

        # fully-connected layer
        # 32*4 input size to account for the downsampled image size after pooling
        # num_classes outputs (for n_classes of image data)
        self.fc1 = nn.Linear(32*4, n_classes)

    # define the feedforward behavior
    def forward(self, x):
        # one conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))

        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        # linear layer 
        x = F.relu(self.fc1(x))

        # final output
        return x

# instantiate and print your Net
n_classes = 20 # example number of classes
net = Net(n_classes)
print(net)

```

### Summary of Guidelines for the Architecture Definition

- Usual architecture: 2-4 `Conv2d` with `MaxPool2d`in-between so that the size is halved; at the end 1-3 fully connected layers with dropout in-between to avoid overfitting.
- Recall that an image has the shape `B x W x H x D`. The bacth size `B` is usually not used during the network programming, but it's there, even though we feed one image per batch!
- In the convolutions:
    - Prefer small 3x3 filters; use odd numbers in any case.
    - Use padding so that the size of the image is preserved! That means taking `padding=floor(F/2)`.
    - Recall the size change formula: `W_out = (W_in + 2P - F)/S + 1`.
- Use `relu()` activation after each convolution, but not after max-pooling.
- If we use a unique decreasing factor in `MaxPool2d`, it's enough defining a unique `MaxPool2d`.
- The typical max-pooling is the one which halvens the size: `MaxPool2d(2,2)`.
- Before entering the fully connected or linear layer, we need to flatten the feature map:
    - In the definition of `Linear()`: we need to compute the final volume of the last feature map set. If we preserved the sized with padding it's easy; if not, we need to apply the formula above step by step.
    - In the `forward()` method: `x = x.view(x.size(0), -1)`; `x.size(0)`is the batch size, `-1` is the rest. 
- If we use `CrossEntropyLoss()`, we need to return the `relu()`/non-activated output; if we return the `log_softmax()`, we need to use the `NLLLoss()`. `CrossEntropy() == log_softmax() + NLLLoss()`.

### Guidelines on Training and Hyperparameter Selection

This section is a summary of the Andrew Ng's notes on **Machine Learning System Design**. All the notes can be found here:

[machine_learning_coursera](https://github.com/mxagar/machine_learning_coursera) `/ 04_MLSystemDesign`

Hyperparameters we can tune:

- batch size: larger values accelerate the computation, but they yield coarser results
- regularization (equivalently, dropout): we can use it to reduce overfitting
- model parameters (equivalently, layers and nodes in the network): we increase them if the model is too simplistic
- learning rate
    - small values lead to slower decrease of the loss
    - too large values lead to oscillation of the loss
    - values to consider `[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]`; we can also use schedule decaying learning rates, because the closer we are to the solution the smaller should be the steps
- number of data samples: if the model is complex enough, more samples improve the result
- number of epochs: same as with the samples: if the model is complex enough, more epochs improve the result, except when we start overfitting (see case 3 below)

Metrics we should observe:

- Training and (cross-) validation loss
- Training and (cross-) validation accuracy

A central point is to detect whether we have:

1. High Bias = We are underfitting, the model is too simplistic to model the data
2. High Variance = We are overfitting, the model is too complex and learns noise from the data
3. Correct trade-off bias-variance = the model correctly learns how to model the data

Depending on each case, we should act differently

**CASE 3**: Correct trade-off bias-variance

- That is the case we should land in.
- We have the correct trade-off when both loss (train & val) decrease with an exponential decay.
- In some cases, when the number of samples/epochs increases, the validation loss might start increasing. That means we are overfitting; several solutions are possible:
    - Save the model with the lowest validation loss
    - Apply **early stopping**
    - Re-define the model with added dropout (i.e., regularization)

**CASE 1: High Bias**: We are underfitting

- Loss curves decay and converge fast to a high value and remain there
- The model is too simplistic to learn the targets
- We need to add more features (i.e., more nodes and layers); **nothing else will help**

**CASE 2: High Variance**: We are overfitting

- Loss curves change (decrease) slowly
- Possible solutions:
    - Decrease complexity of model
    - Increase regularization: dropout in the final linear layers
    - Getting more epochs or more data is likely to help; but before doing that, we should use the samples we have and make sure that the learning curves decrease over time.


## 11. Weight Initialization

Weight initialization can affect dramatically the performance of the training.

If we initialize to 0 all the weights, the backpropagation will have a very hard time to discern in which direction the weights should be optimized. Something similar happens if we initialize the weights with a constant value, say 1.

There are two approaches in weight initialization:

- Xavier Glorot initialization: initialize weights to `Uniform(-n,n)`, with `n = 1 / sqrt(in_nodes)`; i.e., a uniform distirbution in the range given by the number of input nodes in the layer (inverse square root).

- Normal initialization: `Normal(mean=0, std=n)`, with `n = 1 / sqrt(in_nodes)`.

The normal distribution tends to be more performant. We can leave the bias values to 0.

Any time we instantiate a layer in Pytorch, a default Xavier Glorot initialization is applied in the background (we can check that in the code); bias values are also set to a unirform random value.

If we want to train large models, we are encouraged to manually initialize the weights with a normal distribution. To that end, we should use the `apply()` method and loop every layer type to initialize their weights:

```python
# Xavier Glorot Initialization
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # For every Linear layer in a model..
    # We would need to extend that to any type of layer!
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

# Use
model_unirform = Net()
model_uniform.apply(weights_init_uniform)

# Normal Initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''    
    classname = m.__class__.__name__
    # For every Linear layer in a model..
    # We would need to extend that to any type of layer!
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        s = 1.0/np.sqrt(n)
        m.weight.data.normal_(0,std=s)
        m.bias.data.fill_(0)    

# Use
model_normal = Net()
model_normal.apply(weights_init_normal)
```

The notebooks in 

[deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch/) `/weight-initialization`

show how to apply that on a simple MLP that classfies the Fashion-MNIST dataset. There is a helper script which compares models initialized in different ways. Small datasets like the Fashion-MNIST are often used because they are easy and fast to train; thus, we can quickly see how the models behave in few epochs.

## 12. Using the Jetson Nano (CUDA)

Notebooks can be executed on a CUDA device via SSH, e.g., on a Jetson Nano. In order to set up the Jetson Nano, we need to follow the steps in 

    ~/Dropbox/Documentation/howtos/jetson_nano_howto.txt

I created the guide following experimenting myself two very important links:

- [Getting Started with Jetson Nano, Medium](https://medium.com/@heldenkombinat/getting-started-with-the-jetson-nano-37af65a07aab)
- [Getting Started with Jetson Nano, NVIDIA](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro)

In the following, a basic usage guide is provided; for setup, look at the howto file above.

####  Summary of Installation Steps

- Flash SD image
- Create account in Jetson Ubuntu (18): mxagar, pw
- Install basic software
- Create python environment: `env`
- Install DL packages in python environment: Pytorch, Torchvision, PIL, OpenCV, etc.

####  How to Connect to Jetson via SSH

    ssh mxagar@jetson-nano.local

#### Connect to a Jupyter Notebook Run on the Jetson from Desktop

Open new Terminal on Mac: start SSH tunneling and leave it open
    
    ssh -L 8000:localhost:8888 mxagar@jetson-nano.local

Open new Terminal on Mac (or the same works also): connect to jetson-nano & start jupyter
        
    ssh mxagar@jetson-nano.local
    source ~/python-envs/env/bin/activate
    cd /your/path
    jupyter notebook
        or jupyter-lab
        in both cases, look for token
            http://localhost:8888/?token=XXX
                
Open new browser on Mac, go to
     
    http://localhost:8000
    insert token XXX

#### SFTP Access

Currently, I cannot access via SFTP the Jetson Nano. Some configuration is needed, which I didn't have time to go through. As a workaround, I clone and pull/push the repositories to the Jetson directly after connecting via SSH.

#### SCP Access

Copy / Transfer file or folder with SCP:

```bash
# Jetson -> Desktop
scp mxagar@jetson-nano.local:/path/to/file/on/jetson /folder/on/desktop

# Desktop -> Jetson         
scp file.txt mxagar@jetson-nano.local:/path/to/folder/on/jetson
scp -r /local/directory mxagar@jetson-nano.local:/remote/directory
```

## 13. Recursive Neural Networks (RNN)

There is a complete module in my notes on the Udacity Deep Learning Nanodegree dedicated to Recursive Neural Networks; additionally, the Computer Vision Nanodegree has also a module which covers the topic extensively:

- [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity) `/ 03_Advanced_CV_and_DL`
- [deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity) ` / 04_RNN`


### Introduction: Simple RNNs and LSTMs

While CNNs capture spatial relationships, RNNs capture sequential information. The basic way of generating RNNs is using **Simple Recurrent Neurons** or **Elman Networks**, which re-use their previous memory state as input in the next state, as shown in the image:

![Simple RNN](./pics/SimpleRNN.png)

Note that the folded model of the simple RNN neuron is basically like a layer which has three vectors and three mapping matrices:

- `x_t`: input vector at time step `t`.
- `y_t`: output vector at time step `t`.
- `s_t`: memory state at time `t`.
- `W_x`: weight matrix connecting the inputs to the state layer.
- `W_y`: weight matrix connecting the state to the output.
- `W_s`: weight matrix connecting the state from the previous timestep to the state in the following timestep.

Note that we can stack several of such RNN cells or layers; the output of the previous becomes the input for the next. When we define an RNN cell in Pytorch, we specify

- the sizes of the input and hidden/output vectors
- and the number of layers that make up the RNN, i.e., the number of RNN cells that are stacked (typically 1-3).

If we unfold the model and compute the gradients propagating the error in the network, we see that the error derivatives with respect to `W_x` and `W_s` are the summation of the current derivative and all the previous ones.

However, not all previous derivates are considered due to the **vanishing gradient** problem; in practice, the 8-10 previous steps are used. Similarly, to avoid the **exploding gradient** issue, **gradient clipping** is applied: if the gradient exceeds a threshold, it is normalized.

In fact, simple RNNs are not that used because they fall in the **vanishing gradient** problem. Instead, **Long Short-Term Memory (LSTM) units** are usually employed, which were invented by Schmidhuber et al. in 1997 to target specifically the vanishing gradient issue.

The LSTM cells are able to keep for longer periods of time past events: +1000 steps backwards can be considered, not only 8-10. Additionally, with the use of sigmoid and `tanh` activations, LSTM cells can control

- which information to remove
- which to store and
- when to use it
- and when to pass the information to the next stage.

In the following, I will briefly explain the intuition of the LSTM cell without going into details. If interested, Christopher Olah has a great post which explains what's happening inside an LSTM unit: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). 

LSTM cells segregate the memory input/output into two types: 

- short-term memory, which captures recent inputs and outputs.
- and long-term memory, which captures the context.
 
Therefore, in practice we have:

- Three inputs:
  - signal/event: `x_t`
  - previous short-term memory: `h_(t-1)`
  - previous long-term memory : `C_(t-1)`
- Three outputs: 
  - transformed signal or output: `y_t = h_t`
  - current/updated short-term memory: `h_t`
  - current/updated long-term memory: `C_t`

Note that the updated short-term memory is the signal output, too!

All 3 inputs are used in the cell in **4 different and interconnected gates** to generate the 3 outputs; these internal gates are:

- Forget gate: useless parts of previous long-term memory are forgotten, creating a lighter long-term memory.
- Learn gate: previous short-term memory and current event are learned.
- Remember gate: we mix the light long-term memory with forgotten parts and the learned information to form the new long-term memory.
- Use gate: similarly, we mix the light long-term memory with forgotten parts and the learned information to form the new short-term memory.

![LSTM Cell Abstraction](./pics/LSTMs.png)

In summary, the LSTM cell is able to capture the context and the recent items in the fed sequences applying several mappings via four gates internally. A key aspect of the cell is that **it is differentiable**, so we can apply backpropagation and optimize the parameters to minimize the error.

An alternative to LSTMs are the **Gated Recurrent Units (GRUs)**, which appeared later. They simplify the recurrent cell while achieving similar performances.

For more information, see: [Gated Recurrent Units (GRU)](http://www.cs.toronto.edu/~guerzhoy/321/lec/W09/rnn_gated.pdf).

### Defining an LSTM cell in Pytorch

In the following example, the basic usage of an LSTM cell in Pytorch is shown. Input vectors of 4 items map to output vectors of 3 items with one cell. We can pass sequences of vectors, i.e., several vectors arranged in a tensor. One vector can be a word after being transformed into an embedding.

Notes: 

- LSTM units are defined with `nn.LSTM` in Pytorch, and although they are called *units*, they are more like a layer than a neuron, akin to `nn.RNN`; its equivalent would be `nn.Linear`. Additionally, `nn.LSTM` can have several stacked layers inside.
- We can pass one vector after the another in a loop. However, it's more efficient to pass a sequence of vectors together in a tensor. On top of a sequence, we can define batches of sequences. While sequences are usually defined by the application programmer, I'd advise to create batches automatically with the [Pytorch `DataLoader`](https://pytorch.org/docs/stable/data.html) API, as shown in the project [text_generator](https://github.com/mxagar/text_generator).
- When we input a sequence, we get as output a sequence of the same length; the output sequence is composed of hidden memory state vectors. The size of a hidden state vector doesn't need to be the same as the size of an input vector. This can be seen in the project [text_generator](https://github.com/mxagar/text_generator), too; if you'd like more explanations, I encourage you to read [my blog post on that project](https://mikelsagardia.io/blog/text-generation-rnn.html).
- RNNs have many hyperparameters and it can be overwhelming to select the correct starting set. [Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) gives a great collection of hints in his project [char-rnn](https://github.com/karpathy/char-rnn); these are implemented in [text_generator](https://github.com/mxagar/text_generator).

```python
import torch
import torch.nn as nn
from torch.autograd import Variable

# input_dim: number of inputs, length of the input vector, number of units in the input vector
# hidden_dim: number of ouputs, length of the output vector, number of units in the output vector
# n_layers: number of hidden layers used; 1 == the LSTM cell has 1 hidden state
# Note that we pass a SEQUENCE of vectors of dimension input_dim;
# The length of the sequence appears later
input_dim = 4 # example: dimensions of the input word embedding
hidden_dim = 3 # example: number output class categories, if we directly have the output after the LSTM
n_layers = 1 # usually 1 (default) to 3
lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers)
# We can add dropout layers on the outputs of each LSTM layer if we add the argument dropout=drop_p
# drop_p needs to be non-zero; dropout layer is added in all layers except the last one.
# Default: 0
# Also, we can use argument batch_first=True (especially relevant when DataLoaders are used).
# Then, we expect the input size:
# (batch, seq, feature) instead of (seq, batch, feature)
# Similarly, the output size will be
# (batch, seq, out) instead of (seq, batch, out)

# Make a sequence of 5 input vectors of 4 random values each
# Ie., each vector is of dimension 4 (=input_dim)
inputs_list = [torch.randn(1, input_dim) for _ in range(5)]

# Turn input vectors into a tensor with 5 rows of data.
# This is our SEQUENCE.
# Note that the LENGTH of the SEQUENCE is ARBITRARY!
# Add the extra 2nd dimension (1) for batch_size.
# We can also pass each of the 5 vectors one after the other,
# but packing them together is more efficient.
# This seems to be a batch -- but a batch is an additional grouping
# on top of it. IT IS A SEQUENCE.
# Batch size != sequence length.
batch_size = 1
inputs = torch.cat(inputs_list).view(len(inputs_list), batch_size, -1) 

# Size: (number of sequences, batch size, input_dim)
print('inputs size: \n', inputs.size()) # [5, 1, 4]

# Initialize the hidden states: short- and long-term memories
# We have so many as layers we have defined.
# h0: (n_layers, batch_size, hidden_dim)
# c0: (n_layers, batch_size, hidden_dim)
h0 = torch.randn(n_layers, batch_size, hidden_dim)
c0 = torch.randn(n_layers, batch_size, hidden_dim)

# Wrap everything in torch Variable
# Torch Variables are a wrapper around tensors; they represent a node in a graph,
# and they have almost the same operations as tensors.
# To obtain a tensro from a Variable: var.data
# To obtain the gradient tensor of a Variable: var.grad.data
inputs = Variable(inputs)
h0 = Variable(h0)
c0 = Variable(c0)
# get the outputs and hidden state
#output, hidden = lstm(inputs, (h0, c0))
output, (h1, c1) = lstm(inputs, (h0, c0))

# output size: [5, 1, 3]: one output of 3 elements for each of the 5 sequences of 4 elements
# hidden size, (h1, c1): we get the last hidden state; INPUT for the next LSTM
# h1: [1, 1, 3] 
# c1: [1, 1, 3]
```

### Examples

We distinguish different [types of RNN](https://www.javatpoint.com/tensorflow-types-of-rnn), depending on the sequence length for input/output, and example applications:

- One to one: vanilla neural network.
- One to many: image captioning.
- Many to one: sentiment analysis, time series.
- Many to many: part-of-speech tagging, machine translation.

![Types of RNN](./pics/typesRNN.png)

See also [Andrej Karpathy's blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Note: in RNNs, "one" might be one sequence of fixed size.

#### Code / Notebooks

- Simple RNN to forecast time series: [Simple_RNN.ipynb](https://github.com/mxagar/deep-learning-v2-pytorch/blob/master/recurrent-neural-networks/time-series/Simple_RNN.ipynb)
  - A simple RNN is defined which takes one input sequence of 20 values sampled from the `sin()` function and and delivers one sequence of 20 values.
  - The target sequence is defined as the input shifted one step.
  - Therefore, only the last element from the target/output sequence would be really used.

- LSTM RNN with Embeddings to tag Part-of-Speech type of sequences of words: [LSTM_Part-of-Speech_Tagging.ipynb](https://github.com/mxagar/CVND_Exercises/blob/master/2_4_LSTMs/2.%20LSTM%20Training%2C%20Part%20of%20Speech%20Tagging.ipynb)
  - The notebook is very interesting, but the example is very simple: we use a vocabulary of less than 15 words and 3 possible parts-of-speech (noun, verb, determinant).
  - A sequence of arbitrary length (words) is transformed into a sequence of the same length (parts-of-speech).
  - Embeddings are used.
  - Steps:
    - At the beginning, a vocabulary is built: each word gets an index and vice versa.
    - We take a sentence: a sequence of arbitrary length composed of words.
    - The words are converted to indices.
    - The sequence of indices is converted to a sequence of vectors using the embedding; we have a tensor of this size: `[sequence_length, embedding_vector_size]`.
    - The sequence is passed to the LSTM. We need to specify the batch size too, e.g., 1; thus, the tensor is reshaped to `[sequence_length, batch_size=1, embedding_vector_size]`.
    - The output is of size `[sequence_length, batch_size=1, hidden_dimension]`, which is mapped to `[sequence_length, output_classes]` with a linear layer.

- Character-level LSTM to generate text: [Character_Level_RNN_Exercise.ipynb](https://github.com/mxagar/CVND_Exercises/blob/master/2_4_LSTMs/3_1.Chararacter-Level%20RNN%2C%20Exercise.ipynb)
  - It is based on a post by [Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
  - A network is defined to be trained with sequences of characters such that the network is able to predict the next most likely character given the character sequence fed so far. In consequence, the network is able to generate a text character by character.
  - Very interesting example where many topics are touched; could be a blueprint for word-level text generation.
  - Steps:
    - Encoding of text is done: vocabulary (`int2char`, `char2int`) is built and one-hot encoding of text or character sequences is done.
    - Text is properly segmented in mini-batches of a fixed sequence length.
    - Network is defined with LSTMs.
    - Initial memory priming is discussed.
    - Training and text prediction.
    - Hyperparameter tuning is discussed.

- Semantic Embeddings after Word2Vec by Mikolov et al.: [Skip_Grams_Exercise.ipynb](https://github.com/mxagar/deep-learning-v2-pytorch/blob/master/word2vec-embeddings/Skip_Grams_Exercise.ipynb)
  - A semantic embedding is created following the Skip-gram approach by Mikolov et al.
  - This is not an RNN, but embeddings are common in RNNs.
  - A version with optimized training speed which uses negative sampling: [Negative_Sampling_Exercise.ipynb](https://github.com/mxagar/deep-learning-v2-pytorch/blob/master/word2vec-embeddings/Negative_Sampling_Exercise.ipynb)
  - Steps:
    - Text pre-processing: a vocabulary is built with all the unique words in the text and some symbols are replaced by symbol names (`. ->  <PERIOD>`). Additionally, subsampling of the words is done based on their occurrence: a probability of removing a word is defined based on its frequency in the text.
    - Batch generation: we write a generator of batches which receives the text with integer-encoded words and produces sequences of input-target word pairs (encoded as integers).
    - Similarity function
    - SkipGram Model Definition and Training. Note that the training takes very long. This is optimized in the other mentioned notebook.
    - Embedding vector visualization with t-SNE
    - Save the embedding matrix as a dataframe.

- Sentiment analysis RNN with LSTMs: [Sentiment_RNN_Exercise.ipynb](https://github.com/mxagar/deep-learning-v2-pytorch/blob/master/sentiment-rnn/Sentiment_RNN_Exercise.ipynb)
  - This notebook improves the basic sentiment analysis model network without RNNs: [Sentiment Analysis Neural Network with Numpy](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/sentiment-analysis-network).
  - The network receives a batch of reviews. Reviews are tokenized and encoded as integers. The sequence length (number of words per review) is fixed: 200; thus, we either truncate the texts if longer or pad them with 0s on the left.
  - The trained network is able to yield a value 0-1 which denotes the positive (1) or negative (0) sentiment of any text.
  - The efficiency I got with the test split was 96.8%; that's very high!
  - Steps:
    - Load data and pre-process it:
    	- punctuation is removed,
    	- words are tokenized with `split()`
    	- a vocabulary dictionary is built
    	- tokens are encoded as integers
    	- outliers are removed (reviews with length 0)
    	- encoded reviews are converted to a fixed sequence length with truncation or left zero padding
    - Training, validation and test splits are created
    - Data loaders
    - Model definition: Embedding, LSTM, Linear, Dropout, Sigmoid
    - Training
    - Saving and loading model
    - Testing
    - Inference / Prediction function
- [Text Generator Project](https://github.com/mxagar/text_generator)
  - This is a very nice example in which the typical NLP process with RNNs is implemented.
  - My [blog post](https://mikelsagardia.io/blog/text-generation-rnn.html) explains many concepts.

## 14. Recommendations for Hyperparameter Tuning

See the module `03_Advanced_CV_and_DL` of my notes on the Udacity Deep Learning Nanodegree: [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity/blob/main/03_Advanced_CV_and_DL/CVND_Advanced_CV_and_DL.md), Section 6: Hyperparameters.

Also, check [Skorch: A Scikit-Learn Wrapper for Pytorch](https://github.com/skorch-dev/skorch); it should be possible to use Pytorch classifiers in `Pipeline` or `GridSearchCV`.

## 15. Vanilla Inference Pipeline and Artifact

Inference pipelines and artifacts are possible with pytorch.

The folder `./lab/pytorch_inference_pipeline` contains an example of how pipelines work, following these steps:

- A pre-trained ResNet18 is loaded.
- A `Sequential` pipeline is created and `transforms` + model + `Softmax` are packed into it.
- The pipeline is saved as an inference artifact with `torch.jit.script`.
- The inference artifact is loaded.
- An image is loaded, prepared and passed to the pipeline for inference.

To use the example:

```bash
cd  .../lab/pytorch_inference_pipeline
conda activate cvnd
python transforms.py
# ResNet18 is downloaded
# Inference pipeline is saved to disk
# Inference artifact is loaded as well as a test image
# Inference of the image is done
```

The whole inference script `transforms.py` is the following:

```python
import torch
from torchvision import transforms
from torch.nn import Sequential, Softmax
from PIL import Image
import numpy as np

# Get a pre-trained model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.eval()

# Define the inference pipeline
pipe = Sequential(
    # NOTE: for the pipeline to be scriptable with script,
    # you must use a list [256, 256] instead of just one number (256)
    transforms.Resize([256, 256]),
    transforms.CenterCrop([224, 224]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    model,
    Softmax(1)
)

# Save inference artifact using torch.script
scripted = torch.jit.script(pipe)
scripted.save("inference_artifact.pt")

# NOTE: normally we would upload it to the artifact store

# Load inference artifact
pipe_reload = torch.jit.load("inference_artifact.pt")

# Load one example
# NOTE: these operations are usually taken care by the inference
# engine
img = Image.open("dog.jpg")
img.load()
# Make into a batch of 1 element
data = transforms.ToTensor()(np.asarray(img, dtype="uint8").copy()).unsqueeze(0)

# Perform inference
with torch.no_grad():
    logits = pipe_reload(data).detach()

proba = logits[0]

# Transform to class and print answer
with open("imagenet_classes.txt", "r") as f:
    classes = [s.strip() for s in f.readlines()]
print(f"Classification: {classes[proba.argmax()]}")
```

## 16. Cloud Computing with AWS

### 16.1 Launch EC2 Instances

EC2 = Elastic Compute Cloud. We can launch VM instances.

Create an AWS account, log in to the AWS console & search for "EC2" in the services.

Select region on menu, top-right: Ireland, `eu-west-1`. Selecting a region **very important**, since everything is server region specific. Take into account that won't see the instances you have in different regions than the one you select in the menu! Additionally, we should select the region which is closest to us. Not also that not all regions have the same services and the service prices vary between regions!

Press: **Launch Instance**.

Follow these steps:

1. Choose an Amazon Machine Image (AMI) - An AMI is a template that contains the software configuration (operating system, application server, and applications) required to launch your instance. I looked for specific AMIs on the search bar (keyword "deep learning") and selected `Deep Learning AMI (Amazon Linux 2) Version 61.3` and `Deep Learning AMI (Amazon Linux 2) Version 61.3` for different instances. Depending on which we use, we need to install different dependencies.

2. Choose an Instance Type - Instance Type offers varying combinations of CPUs, memory (GB), storage (GB), types of network performance, and availability of IPv6 support. AWS offers a variety of Instance Types, broadly categorized in 5 categories. You can choose an Instance Type that fits our use case. The specific type of GPU instance you should launch for this tutorial is called `p2.xlarge` (P2 family). I asked to increase the limit for EC2 in the support/EC2-Limits menu option to select `p2.xlarge`, but they did not grant it to me; meanwhile, I chose `t2.micro`, elegible for the free tier.

3. Configure Instance Details - Provide the instance count and configuration details, such as, network, subnet, behavior, monitoring, etc.

4. Add Storage - You can choose to attach either SSD or Standard Magnetic drive to your instance. Each instance type has its own minimum storage requirement.

5. Add Tags - A tag serves as a label that you can attach to multiple AWS resources, such as volumes, instances or both.

6. Configure Security Group - Attach a set of firewall rules to your instance(s) that controls the incoming traffic to your instance(s). You can select or create a new security group; when you create one:
    - Select: Allow SSH traffic from anywhere
    - Then, when you launch the instance, **you edit the security group later**
    - We can also select an existing security group

7. Review - Review your instance launch details before the launch.

8. I was asked to create a key-pair; I created one with the name `face-keypoints` using RSA. You can use a key pair to securely connect to your instance. Ensure that you have access to the selected key pair before you launch the instance. A file `face-keypoints.pem` was automatically downloaded.

More on [P2 instances](https://aws.amazon.com/ec2/instance-types/p2/)

Important: Edittting the security group: left menu, `Network & Security` > `Security Groups`:

- Select the security group associated with the created instance (look in EC2 dashboard table)
- Inbound rules (manage/create/add rule):
    - SSH, 0.0.0.0/0, Port 22
    - Jupyter, 0.0.0.0/0, Port 8888
    - HTTPS (Github), 0.0.0.0/0, Port 443
- Outbound rules (manage/create/add rule):
    - SSH, 0.0.0.0/0, Port 22
    - Jupyter, 0.0.0.0/0, Port 8888
    - HTTPS (Github), 0.0.0.0/0, Port 443

If we don't edit the security group, we won't be able to communicate with the instance in the required ports!

**Important: Always shut down / stop all instances if not in use to avoid costs! We can re-start afterwards!**. AWS charges primarily for running instances, so most of the charges will cease once you stop the instance. However, there are smaller storage charges that continue to accrue until you **terminate** (i.e. delete) the instance.

We can also set billing alarms.

### 16.2 Connect to an Instance

Once the instance is created, 

1. We `start` it: 

    - EC2 dashboard
    - Instances
    - Select instance
    - Instance state > Start

2. We connect to it from our local shell

```bash
# Go to the folder where the instance key pem file is located
cd .../project
# Make sure the pem file is only readable by me
chmod 400 face-keypoints.pem
# Connect to instance
# user: 'ec2-user' if Amazon Image, 'ubuntu' if Ubuntu image
# Public IP: DNS or IP number specified in AWS EC2 instance properties
# ssh -i <pem-filename>.pem <user>@<public-IP>
ssh -i face-keypoints.pem ec2-user@3.248.188.159
# We need to generate a jupyter config file
jupyter notebook --generate-config
# Make sure that
# ~/.jupyter/jupyter_notebook_config.py
# contains 
# c.NotebookApp.ip = '*'
# Or, alternatively, directly change it:
sed -ie "s/#c.NotebookApp.ip = 'localhost'/#c.NotebookApp.ip = '*'/g" ~/.jupyter/jupyter_notebook_config.py
# Clone or download the code
# Note that the SSH version of the repo URL cannot be downloaded;
# I understand that's because the SSH version is user-bound 
git clone https://github.com/mxagar/P1_Facial_Keypoints.git
# Go to downloaded repo
cd P1_Facial_Keypoints
# When I tried to install the repo dependencies
# I got some version errors, so I stopped and
# I did not install the dependencies.
# However, in a regular situation, we would need to install them.
# Also, maybe:
# pip install --upgrade setuptools.
sudo python3 -m pip install -r requirements.txt
# Launch the Jupyter notebook without a browser
jupyter notebook --ip=0.0.0.0 --no-browser
# IMPORTANT: catch/copy the token string value displayed:
# http://127.0.0.1:8888/?token=<token-string>
```

3. Open our local browser on this URL, composed by the public IP of the EC2 instance we have running and the Jupyter token:

```
http://<public-IP>:8888/?token=<token-string>
```

### 16.3 Pricing

Always stop & terminate instances that we don't need! Terminates erases any data we have on the instance!

[Amazon EC2 On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/)

## 17. Beyond Classification: Object Detection and Semantic Segmentation

The Udacity Computer Vision Nanodegree covers in the 3rd module the topics of Object Detection and Semantic Segmentation, among others.

See also [detection_segmentation_pytorch](https://github.com/mxagar/detection_segmentation_pytorch).


## Appendix: Tips and Tricks

### Number of Model Parameters

```python
# Get number of model parameters
def get_num_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
```

### Torch Summary

The library [torchsummary](https://pypi.org/project/torchsummary/) provides a very nice summary of the network, similar to what Keras does:

```python
from torchsummary import summary
model = Net()
summary(model)
```

### Running Average Loss

We can either sum the loss and the divide it by the number of batches or compute the running average:

```python
# Running Average
avg_loss = 0

for e in range(epochs):
    # ...
    avg_loss += (1.0 / (batch_id + 1)) * (loss.item() - avg_loss)

```

### Export Conda and Pip Environments

```bash
conda env export -f environment.yaml
pip freeze > requirements. txt
```

### Apply Good Code and Reproducibility Guidelines

- Use randomness seeds and controlled weight initialization to allow reproducibility.
- Embed code in classes and create scripts.
- Logging.
- Follow PEP8 guidelines.
- [Ultimate Guide to Python Debugging](https://towardsdatascience.com/ultimate-guide-to-python-debugging-854dea731e1b)
- Try [Pytorch profiling](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### Imbalaced Datasets

Imbalanced datasets are those classification datasets that contain significantly different numbers of instances for each class. Over- or under-sampling techniques need to be applied if we want to avoid introducing class bias.

Check, for instance: [imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler).

### Use `Sequential` to Be Cleaner

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(56*56*32, 133)
        self.fc2 = nn.Linear(1024, 133)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
```

### Improving the Training: Learning Rate Scheduler and Optimization Algorithms

The section [Guidelines on Training and Hyperparameter Selection](#Guidelines-on-Training-and-Hyperparameter-Selection) provides some tips on how to determine whether the network is training properly or not.

One additional way of improving the training is by using a learning scheduler, as explained in the [docu of optim](https://pytorch.org/docs/stable/optim.html). An example scheduler could be:

```python
scheduler = optim.lr_scheduler.StepLR(optimizer_scratch, step_size=100, gamma=0.9)
```

Finally, check also:

- [Optimization algorithms by Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/)
- [Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

### Print Memory Usage During Training

```python
print("use_cuda: ",use_cuda," -> ", torch.cuda.get_device_name(0))
print('Memory Usage:')
print('\tAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('\tCached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
```

## Appendix: Lab - Example Projects

The following files give a very nice overview of how Pytorch is used for image classification:

- `fc_model.py`: complete pipeline of a fully connected notebook for image classification
- `helper.py`: visualization.
- `Part 7 - Loading Image Data.ipynb`: dealing with custom datasets.
- `Part 8 - Transfer Learning.ipynb`: transfer learning example with a CNN backbone for image classification.

However, many applications go beyond those use cases. To that end, I will collect in the folder `./lab` more blueprints/examples of different applications.

Please, go to the `./lab` folder are read the `README.md` there to get more information.

## Appendix: Important Links

- [Deep Tutorials for PyTorch](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch)
- [Understanding LSTM Networks, by Chris Olah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Exploring LSTMs, by Edwin Chen](http://blog.echen.me/2017/05/30/exploring-lstms/)
- [Karpathy's Lecture: Recurrent Neural Networks, Image Captioning, LSTM](https://www.youtube.com/watch?v=iX5V1WpxxkY)
- [Skorch: A Scikit-Learn Wrapper for Pytorch](https://github.com/skorch-dev/skorch): Use Pytorch classifiers in `Pipeline` or `GridSearchCV`.