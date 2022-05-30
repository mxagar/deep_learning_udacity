# Udacity Deep Learning Nanodegree: Personal Notes

These are my personal notes taken while following the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

The nanodegree is composed of six modules:

1. Introduction to Deep Learning
2. Neural Networks and Pytorch Guide
3. Convolutonal Neural Networks (CNN)
4. Recurrent Neural Networks (RNN)
5. Generative Adversarial Networks (GAN)
6. Deploying a Model

Each module has a folder with its respective notes.

Additionally, note that:
- I made many hand-written nortes, which I will scan and push to this repostory.
- I forked the Udacity repository for the exercisesl [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch); all the material and  notebooks are there.

## Practical Installation Notes

I basically followed the installation & setup guide from [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch), which can be summarized with the following commands:

```bash
# Create new conda environment to be used for the nanodegree
conda create -n dlnd python=3.6
conda activate dlnd
conda install pytorch torchvision -c pytorch
conda install pip
#conda install -c conda-forge jupyterlab
# Go to the folder where the Udacity DL exercises are cloned, after forking the original repo
cd ~/git_repositories/deep-learning-v2-pytorch
pip install -r requirements.txt
```

Mikel Sagardia, 2022.  
No guarantees.