# Udacity Deep Learning Nanodegree: Introduction

These are my personal notes taken while following the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

The nanodegree is composed of six modules:

1. Introduction to Deep Learning
2. Neural Networks
3. Convolutonal Neural Networks (CNN)
4. Recurrent Neural Networks (RNN)
5. Generative Adversarial Networks (GAN)
6. Deploying a Model

Each module has a folder with its respective notes. This folder is the one of the **first module**: Introduction to Deep Learning.

Additionally, note that:
- I made many hand-written nortes, which I will scan and push to this repostory.
- I forked the Udacity repository for the exercisesl [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch); all the material and  notebooks are there.

## Overview of Contents

1. Introduction to Deep Learning
	- Lesson 4: Anaconda
	- Lesson 5: Applying Deep Learning
		- Style Transfer
		- DeepTraffic by Lex Fridman
		- Flappy Bird
		- Books to read
	- Lesson 6: Jupyter Notebook
	- Lesson 7: Matrix Math and Numpy Refresher
2. Neural Networks
	- Lesson 1: Introduction to Neural Networks

# 1. Introduction to Deep Learning

## Lesson 4: Anaconda

Packae distribution for data science.

Miniconda = Anaconda - Preinstalled Packages.

`pip`: official python package installation and management; it is installed automatically.
- pip is for general purpose packages, not only python
conda is for data science packages, often precompiled
so both pip and conda are used

Other concepts:
- virtual environments
- conda packages
 -best practices

## Lesson 5: Applying Deep Learning

### Style Transfer

```bash
cd ~/git_repositories/foreing
git clone https://github.com/lengstrom/fast-style-transfer
conda create -n style-transfer python=3
conda activate style-transfer
conda install nomkl
conda install numpy scipy pandas tensorflow pillow
conda remove mkl mkl-service
conda install pip
pip install moviepy
python -c "import imageio; imageio.plugins.ffmpeg.download()"
# if RuntimeError:
# pip install imageio-ffmpeg

# Download checkpoints of trained networks
# 	wave.ckpt
# 	rain-princess.ckpt
#	...
#	Rain Princesss, by Leonid Afremov
#	La Muse, by Pablo Picasso
#	Udnie by Francis Picabia
#	Scream, by Edvard Munch
#	The Great Wave off Kanagawa, by Hokusai
#	The Shipwreck of the Minotaur, by J.M.W. Turner

# Take your image (not that big size!) and evaluate with a trained model of your choise (wave, rain-princess, etc)
python evaluate.py --checkpoint ./rain-princess.ckpt --in-path <path_to_input_file> --out-path ./output_image.jpg

python evaluate.py --checkpoint ./rain-princess.ckpt --in-path house.jpeg --out-path ./output_image.jpg

python evaluate.py --checkpoint ./wave.ckpt --in-path house.jpeg --out-path ./output_image_wave.jpg
```

### DeepTraffic by Lex Fridman

[https://github.com/lexfridman/deeptraffic](https://github.com/lexfridman/deeptraffic)

### Flappy Bird

[https://github.com/yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)

Agent oplaying game after Reinforcement Learning has been applied

### Books to read
	
- Grokking Deep Learning - Andrew Trask
- Neural Networks and Deep Learning - Michael Nielsen: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)
- The Deep Learning Textbook - Ian Goodfellow

## Lesson 6: Jupyter Notebook

Literate Programming: Proposed by Donald Knuth in 1984: documentation written as a narrative alongside the code.

It started with IPython and evolved as a web-based notebook in 2014.
IPython is an interactive shell with syntax highlighting & code completion.

Architecture:
- Kernel: it executes the code. At the beginning it was IPython, not anymore, since many language-kernels are available, eg: R, Julia, ...
- Notebook server
	- connected to the kernel and the browser
	- it stores the notebook as a JSON with .ipynb ending
	- it sends code to the kernel and receives it back
	- it sends the output to the browser
- Broweser interface: the user interacts with it
- IMPORTANT NICE THING OF THIS ARCHITECTURE: we can start a server on a machine and connect with our browser to it from anywhere! Example: server & kernel on Amazon EC2, notebook brower on our computer

Installation & Use: covered in `python_manual.txt`


## Lesson 7: Matrix Math and Numpy Refresher

- Scalars, vectors, matrices, tensors: tensors can be 2+D matrices, an element in a matrix tensors could be a matrix/tensor itself
- Numpy

# 2. Neural Networks

## Lesson 1: Introduction to Neural Networks

The content is mostly recorded in the hand-written notes.
There is overlap with the CVND = Udacity Computer Vision Nanodegree (overlap in videos, notebooks, etc.).

List of concepts:

- Classification
- Perceptron
- Perceptron as Logical Operators
- Perceptron model optimization: Perceptron algorithm (mis-classified points added to the model parameters after scaling with learning rate)
- Error
- Discrete vs Continuous Outputs: We need to hava a continuous differentiable error
- Sigmoid Function: continuous output compressed to `(0,1)`
- Softmax Function: multi-class sigmoid
- One-hot encoding
- Maximum Likelyood: the best model has the highest maximum likelihood: products of the predicted data-point probabilities of the correct value; however, in practice the `log` is used to avoid products of small numbers!
- Cross Entropy: a way of interpreting the cross-entropy loss is the maximum likelihood product computed as sums of `logs`.
	- That way, the maximim likelihood is the probability of all the classes, and the cross-entropy is the error. The higher the probability, the lower the error!
	- Another way of interpreting the cross-entropy is the distance error from a discrete vector to our continuous probabilities: `CE([1,1,0],[0.8,0.7,0.1]) = 0.69`
- Error Function
- Graident Descend
- Comparison: Perceptron Algorithm vs Gradient descend
- Nonlinear Models: Multi-layer Perceptrons = Neural Networks
- Feedforward
- Backpropagation

Interesting Jupyter Notebooks: [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch)

- `intro-neural-networks/GradientsDescend`: Gradient descend applied to the same dataset as before.
- `intro-neural-networks/StudentAdmissions`
	- Gradient descend applied to a linear model
	- Dataset: student admission data: 3D data (test result, GPA, class rank percentile), converted to one-hot 6D
	-One-hot encoding is done in pandas using 2 lines: Rank 0-4 -> rank_i 0/1 for i 1-4



