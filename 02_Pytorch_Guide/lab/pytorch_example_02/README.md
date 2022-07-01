# CIFAR-10 Dataset: Manual CNN & Transfer Learning on Jetson Nano (CUDA)

This project performs classification on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The following concepts are explored:

- Manual definition of a CNN similar to LeNet to test classification.
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

