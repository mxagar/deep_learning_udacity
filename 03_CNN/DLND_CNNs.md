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
2. Cloud Computing (and GPU Workspaces)
3. Transfer Learning
4. Weight Initialization
5. Autoencoders
6. Style Transfer
7. Project: Dog-Breed Classifier
8. Deep Learning for Cander Detection
9. Jobs in Deep Learning
10. Project: Optimize Your GitHub Profile



## 1. Convolutional Neural Networks



## 2. Cloud Computing (and GPU Workspaces)



## 3. Transfer Learning



## 4. Weight Initialization



## 5. Autoencoders



## 6. Style Transfer



## 7. Project: Dog-Breed Classifier



## 8. Deep Learning for Cander Detection

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


