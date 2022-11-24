# Udacity Deep Learning Nanodegree: Personal Notes

These are my personal notes taken while following the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

The nanodegree is composed of six modules:

1. [Introduction to Deep Learning](01_Intro_Deep_Learning)
2. [Neural Networks and Pytorch/Keras Guides](02_Neural_Networks)
3. [Convolutional Neural Networks (CNN)](03_CNN)
4. [Recurrent Neural Networks (RNN)](04_RNN)
5. [Generative Adversarial Networks (GAN)](05_GAN)
6. [Deploying a Model with AWS SageMaker](06_Deployment)

Additionally, I have added an extra module/subfolder which I will extend with *new* architectures and applications that appeared post 2018: [Extra](07_Extra).

Each module has a folder with its respective notes; **you need to go to each module folder and follow the Markdown file in them**.

Finally, note that:

- I have also notes on the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891) in my repository [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity); that MOOC is strongly related and has complementary material.
- In addition to the [Pytorch guide](02_Pytorch_Guide), I have a [Keras guide](02_Keras_Guide); both condense the most important features of both frameworks. Currently, the Pytorch guide is more detailed.
- I have many hand-written notes you can check, too (see the PDFs).
- The exercises are commented in the Markdown files and linked to their location; most of the exercises are located in other repositories, originally forked from Udacity and extended/completed by me:
	- [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch)
	- [CVND_Exercises](https://github.com/mxagar/CVND_Exercises)
	- [DL_PyTorch](https://github.com/mxagar/DL_PyTorch)
	- [CVND_Localization_Exercises](https://github.com/mxagar/CVND_Localization_Exercises)
	- [sagemaker-deployment](https://github.com/mxagar/sagemaker-deployment)

## Projects

Udacity requires the submission of a project for each module; these are the repositories of the projects I submitted:

1. Predicting Bike Sharing Patterns with Neural Networks Written from Scratch with Numpy: [project-bikesharing](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/project-bikesharing).
2. Dog Breed Classification with Convolutional Neural Networks (CNNs) and Transfer Learning: [project-dog-classification](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/project-dog-classification).
3. Text Generation: TV Script Creation with a Recurrent Neural Network (RNN): [text_generator](https://github.com/mxagar/text_generator).
4. Face Generation.
5. Deployment of a Sentiment Analysis Model.

## Practical Installation Notes

I basically followed the installation & setup guide from [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch), which can be summarized with the following commands:

```bash
# Create new conda environment to be used for the nanodegree
conda create -n dlnd python=3.6
conda activate dlnd
conda install pytorch torchvision -c pytorch
conda install pip

# Go to the folder where the Udacity DL exercises are cloned, after forking the original repo
cd ~/git_repositories/deep-learning-v2-pytorch
pip install -r requirements.txt
```

Mikel Sagardia, 2022.  
No guarantees.

If you find this repository helpful and use it, please link to the original source.
