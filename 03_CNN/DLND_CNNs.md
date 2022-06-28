# Pytorch Guide

These are my personal notes taken while following the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

The nanodegree is composed of six modules:

1. Introduction to Deep Learning
2. Neural Networks and Pytorch Guide
3. Convolutonal Neural Networks (CNN)
4. Recurrent Neural Networks (RNN)
5. Generative Adversarial Networks (GAN)
6. Deploying a Model

Each module has a folder with its respective notes. This folder is the one of the **third module**: Convolutional Neural Networks.

Additionally, note that:

- I made many hand-written nortes, which I will scan and push to this repostory.
- I forked the Udacity repository for the exercisesl [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch); all the material and  notebooks are there.


## Overview of Contents

1. Convolutional Neural Networks
2. Cloud Computing (and GPU Workspaces): See the CVND
3. Transfer Learning
4. Weight Initialization
5. Autoencoders
6. Style Transfer
7. Project: Dog-Breed Classifier
8. Deep Learning for Cander Detection
9. Jobs in Deep Learning
10. Project: Optimize Your GitHub Profile

## 1. Convolutional Neural Networks

Many of the concepts in this module are covered in the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891). See my notes on it, especially the module 1: [Introduction to Computer Vision](https://github.com/mxagar/computer_vision_udacity).

In the following, I very briefly collect the terms of known concepts and extend only in new material.

### 1.1 Applications of CNNs

Some applications and links:

- [WaveNet](https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio): convolutions on the sound stream are applied to synthesize speech. It can be used to generate music, too.
- Text classification; RNNs are more typical for text, though.
- Image classification.
- Reinforcement learning for playing games: Games can be learned from images.
- AlphaGo also used CNNs underneath.
- Traffic sign classification: [German Traffic Sign dataset in this project](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset); check out this [Github repo](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)
- [Depth map prediction from a single image](https://cs.nyu.edu/~deigen/depth/).
- [Convert images into 3D maps for blind people](https://www.businessinsider.com/3d-printed-works-of-art-for-the-blind-2016-1).
- [Breast cancer detection](https://ai.googleblog.com/2017/03/assisting-pathologists-in-detecting.html).
- [FaceApp](https://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/): change your face expression.

### 1.2 CNNs: Introductory Concepts

Many of the concepts in this module are covered in the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891). See my notes on it, especially the module 1: [Introduction to Computer Vision](https://github.com/mxagar/computer_vision_udacity). Here, I very briefly list the concepts discussed in this section:

- MNIST dataset: what it is, sizes, etc.
- Image normalization: from 255 to 1; it improves backpropagation.
- Flattening of a 2D matrix to feed patches into fully connected networks that end up predicting class scores.
- Hidden layers: google for papers that suggest concrete numbers.
- Loss functions: Cross-Entropy loss for classification.
- Softmax function: multi-class classification.
- ReLU activation.
- Train/Test split.
- Pytorch: `CrossEntropy() == log_softmax() + NLLLoss()`.

### 1.3 MNIST MLP Exercise

The exercise notebooks are in here:

[deep-learning-v2-pytorch/tree/master/convolutional-neural-networks/mnist-mlp](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/convolutional-neural-networks/mnist-mlp)

Basically, an MLP is defined to classify MNIST digits; the code is very similar to the Section 3 in the `02_Pytorch_Guide` module of this repository.

```python

import torch.nn as nn
import torch.nn.functional as F

## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Definition of hidden nodes
        hidden_1 = 512
        hidden_2 = 256

        # Linear: W(input,output), B(1,output) -> x*W + B
        # W: model.fc1.weight
        # B: model.fc1.bias        
        # First layer: input
        # 28*28 -> hidden_1 hidden nodes
        self.fc1 = nn.Linear(784, hidden_1)
        
        # Second layer: hidden
        # hidden_1 -> hidden_2
        self.fc2 = nn.Linear(hidden_1, hidden_2)

        # Output layer: units = number of classes
        # hidden_2 -> 10
        self.fc3 = nn.Linear(hidden_2, 10)
        
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        # Flatten image input
        x = x.view(-1, 28 * 28)
        # Add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        x = self.fc3(x)
        # Final tensor should have a size batch_size x units: 64 x 10
        # dim=1: sum across columns for softmax
        x = F.softmax(x, dim=1) # alternative: x = self.softmax(x)
        
        return x

# initialize the NN
model = Net()
print(model)

## Specify loss and optimization functions
from torch import optim

# specify loss function
criterion = nn.NLLLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### 1.4 Validation

We should split out dataset in 3 exclusive groups:

1. Training split: to train.
2. Validation split: to test how well the model generalizes and to choose between hyperparameters.
3. Test split: to evaluate the final model performance.

The training is performed with the training split, while we continuously (e.g., after each epoch) check the validation loss of the model so far. If the model starts overfitting, the training loss will decrease while the validation loss will start increasing. The idea is to save the weights that yield the smallest validation loss. We can do it with early stopping or just saving the weights of the best epoch.

![Cross-validation](./pics/cross_validation.png)

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


## 2. Cloud Computing (and GPU Workspaces)

See the repository of the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891):

[computer_vision_udacity(https://github.com/mxagar/computer_vision_udacity) / `02_Cloud_Computing`.

## 3. Transfer Learning



## 4. Weight Initialization



## 5. Autoencoders



## 6. Style Transfer



## 7. Project: Dog-Breed Classifier



## 8. Deep Learning for Cancer Detection

```
# Intro

	Melanoma cuases 10.000 deaths/year in USA
		Traffic accidetnt 40.000
	Four stages
		1. Superficial
		4. It has reached blood vessels (it can happen within a year)
	Survival 5 years
		Stage 1: 100%
		Stage 4: 20%
		Therefore, early detection is very important!
	Very hard to distinguish a melanoma from a mole (lunar)
		dermatologists are very trained

# Dataset

	more than 20.000 labelled images collected
	there are more than 2.000 of disease classes...
		they were distilled to 757 classes
		many of the 2000 were duplicates, had missspellings, etc
	melanoma, the worst and most lethal, is one of them
	very challenging classification

# Network & Training
	
	Inception-v3, from Google
	They traine dit 2x
		once with randomly initialized weights
		once with previoulsy trained weights - network pretrained with regular images, cats & dogs, etc
			surprisingly, it led to better results!
			although the training images are different from the skin images,
			apparently significant features and structures are learned when observing the world

# Validation

	Dataset was carefully cleaned
	They wanted to remove duplicates, remove yellow scaling markers, etc
	They wanted/needed to have clean and correct training and validation/test sets, independent

	After training, they achieved a better accuracy than a dermatologist
		CNN accuracy: 72%
		Dermatologist 1 accuracy: 65.6%
		Dermatologist 2 accuracy: 66.0%

# Precision, Recall, Accuracy

	See handwritten notes.

	Confusion matrix with
		actual truth: +, -
		predicted: +, -
		quadrants:
			true positive, TP
			false positive, FP -> type I error: conservative error
			TN
			FN -> type II error: AVOID!

	Ratios
		precision = TP / (TP + FP)
		recall = true positive rate = TP / (TP + FN)
		accuracy = (TP + TN) / All
		true negative rate = TN / (TN + FP)
		false positive rate = FP / (TN + FP)

	Sensitivity = Recall
		of all sick people, how many did we diagnose sick?
	Specificity = true negative rate
		of all healthy people, how many did we diagnose healthy?

# Detection Score Threshold

	See handwritten notes.
	
	Network output: P = probability of malignant
	Where should we put the threshold?
	Set both distributions on same axis
		bening
		malignant
	In the medical context, we should take a conservative threshold (type I error) that eliminates all FN (type II error)

# ROC Curve = Receiver Operating Characteristic
	
	See handwritten notes.
	
	Plot that illustrates the diagnostic ability of a binary classifier.
	There are several ROC curves
		false positive rate vs true positive rate
		sesitivity vs specificity -> common in medical context

	Computed this way
		P = probability of malignant
		Set both distributions of malignant and bening on this axis
		Sweep from left to right threshold and compute the pair
			false positive rate vs true positive rate
			sesitivity vs specificity
		and plot

	Area Under the Curve (AUC) should be as close as possible to 1.0

	Several cases analyzed in the handwritten notes

	Interpretation
		Specificity: high values make the overall cost more efficient
			-> related to type I error
		Sensitivity: it's critical and should be as high as possible 
			-> related to type II error

	Most of the tested 25 dermatologists were below the ROC curve
		the spread is quite large
		it seems some have a lower sensitivity: is it because it is costly for the insurance companies to run more tests?

# Visualization

	See handwritten notes.
	
	t-SNE
	Sesibility analysis
		change samples and observe change in class output to understand what is the net looking at
		-> heatmaps, saliency maps

# Confusion Matrix	

	See handwritten notes.
	
	If we have seveal classes, they tell as the probability of patients having A while they're diagnosed with B

	The confusion matrix should as close as possible to the identity, and in any case the less sparse possible

	The two tested dermatologists had a more sparse confusion matrix than teh network 

```


## 9. Jobs in Deep Learning



## 10. Project: Optimize Your GitHub Profile


