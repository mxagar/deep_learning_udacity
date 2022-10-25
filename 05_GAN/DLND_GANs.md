# Geneartive Adversarial Networks (GAN)

These are my personal notes taken while following the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

The nanodegree is composed of six modules:

1. Introduction to Deep Learning
2. Neural Networks and Pytorch Guide
3. Convolutional Neural Networks (CNN)
4. Recurrent Neural Networks (RNN)
5. Generative Adversarial Networks (GAN)
6. Deploying a Model

Each module has a folder with its respective notes. This folder is the one of the **fifth module**: Generative Adversarial Networks.

Additionally, note that:

- I made many hand-written notes; check the PDFs.
- I forked the Udacity repository for the exercises [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch); all the material and notebooks are there.

## Overview of Contents

- [Geneartive Adversarial Networks (GAN)](#geneartive-adversarial-networks-gan)
  - [Overview of Contents](#overview-of-contents)
  - [1. Generative Adversarial Networks (GANs)](#1-generative-adversarial-networks-gans)
    - [1.1 Applications of GANs](#11-applications-of-gans)
    - [1.2 How GANs Work](#12-how-gans-work)
  - [X. Diffusion Models](#x-diffusion-models)
  - [X. NERFs](#x-nerfs)

## 1. Generative Adversarial Networks (GANs)

This section is taught by Ian Goodfellow himself! Goodfellow invented the Generative Adversarial Networks (GANs).

### 1.1 Applications of GANs

Some applications of GANs:

- Generate images from text: [StackGAN](https://arxiv.org/abs/1612.03242), [Dall-E](https://arxiv.org/pdf/2102.12092.pdf), [Stable Diffusion](https://arxiv.org/pdf/2112.10752.pdf), etc.
- Generate images sketches: [iGAN](https://github.com/junyanz/iGAN)
- Transform images into cartoons: [CartoonGAN](https://video.udacity-data.com/topher/2018/November/5bea23cd_cartoongan/cartoongan.pdf)

The first generates images that have been never seen by sampling in a distribution of all possible images.

The last two use a so called **Pix2Pix** approach in which an image in one domain is transformed into another domain; aka. *image translation*. It is possible to train those models in an unsupervised way, as Facebook researchers have shown with a model that transforms face photos into cartoons.

Another example is [CycleGAN](https://junyanz.github.io/CycleGAN/): unpaired image-to-image translation, i.e., they can transform an image from one domain to another in the absence of paired examples; for instance: a horse becomes a zebra. The training is unsupervised.

Some other applications of *Pix2Pix*:

- Building blueprints to finished building.
- Drawings to images.
- Photos to cartoons.
- Day images to night images.
- Adversarial examples that fool machines but not humans (captchas).
- Apple: eye image sketches to realistic eyes; then they trained a model to detect where the user is looking.

In the last example we see that GANs can be used to generate synthetic training sets.

GANs are not limited to the visual domain; we can use them to imitate anything:, e.g., Reinforcement Learning actions. 

### 1.2 How GANs Work



## X. Diffusion Models

- [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU&list=LL)
- [Introduction to Diffusion Models for Machine Learning](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)


## X. NERFs

- [NeRFs: Neural Radiance Fields - Paper Explained](https://www.youtube.com/watch?v=WSfEfZ0ilw4)
- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [Jon Barron - Understanding and Extending Neural Radiance Fields](https://www.youtube.com/watch?v=HfJpQCBTqZs)