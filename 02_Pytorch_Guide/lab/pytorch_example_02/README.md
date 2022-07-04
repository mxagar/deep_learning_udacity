# CIFAR-10 Dataset: Manual CNN & Transfer Learning on Jetson Nano (CUDA)

This project performs classification on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 dataset consists of 60000 32x32 colour images divided into 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The following concepts are explored:

- Manual definition of a CNN similar to LeNet to test classification.
- Data augmentation.
- Three splits: train, validation and test.
- Transfer learning for a better classification.
- Training on a CUDA device, Jetson Nano per SSH.

In order to use these examples, we need to install Pytorch in out environment (eg., by using Anaconda):

```bash
conda create -n myenv python=3.6
source activate myenv
conda install opencv-python matplotlib numpy pillow jupyter scipy pandas
conda install pytorch torchvision -c pytorch
# I had some issues with numpy and torch
pip uninstall numpy
pip uninstall mkl-service
pip install numpy
pip install mkl-service
```

For a more detailed guide on Deep Learning and its implementation and use with Pytorch, have a look at my complete repository on that topic:

[deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity)

I have used several source files to compile this project, mainly from Udacity:

[deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch)

- `/convolutional-neural-networks/cifar-cnn/`
- `/transfer-learning/`

## Files and content

This repository contains all the code in one notebook:

`cifar10_CNN_transfer_learning.ipynb`

The notebook has the following sections:

1. Imports and Test for CUDA
2. Load and Augment the Data
	- 2.1 Visualize the Dataset
3. Define the CNN Manually
	- 3.1 Loss Function and Optimizer
4. Train the Network
	- 4.1 Load the Best Network
5. Evaluate the Network
	- 5.1 Visualize Some Results
6. Remote Execution on CUDA Device, Jetson Nano
7. Transfer Learning
	- 7.1 Load Backbone and Modify Classifier
	- 7.2 Train
	- 7.3 Evaluate
	- 7.4 Visualize Some Results

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to copy and re-use my code; please, reference my authorship and, especially, the sources I have used.
