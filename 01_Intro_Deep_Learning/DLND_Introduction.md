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

- I made many hand-written notes; check the PDFs.
- I made many hand-written notes; check the PDFs.
- I forked the Udacity repositories for the exercises; most the material and notebooks are there:
  - [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch)
  - [DL_PyTorch](https://github.com/mxagar/DL_PyTorch)
  - [sagemaker-deployment](https://github.com/mxagar/sagemer-deployment)

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

