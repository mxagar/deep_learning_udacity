# Tensorflow/Keras Guide

I completed this guide after following these two courses:

- [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) offered by IBM & Coursera.
- [Complete Tensorflow 2 and Keras Deep Learning Bootcamp](https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/) offered by José Marcial Portilla on Udemy.

However, I decided to locate this guide in the repository where I have my notes on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) to keep all Deep Learning material centralized in one location.

My repositories related to source courses are:

- [machine_learning_ibm/05_Deep_Learning](https://github.com/mxagar/machine_learning_ibm/tree/main/05_Deep_Learning)
- [data_science_python_tools/19_NeuralNetworks_Keras](https://github.com/mxagar/data_science_python_tools/tree/main/19_NeuralNetworks_Keras).

Mikel Sagardia, 2022.  
No guarantees.

## Table of Contents

- [Tensorflow/Keras Guide](#tensorflowkeras-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction: Basics](#1-introduction-basics)
    - [1.1 Dropout as Regularization](#11-dropout-as-regularization)
    - [1.2 Early Stopping (as Regularization)](#12-early-stopping-as-regularization)
    - [1.3 Tensorboard](#13-tensorboard)
    - [1.4 Notes and Best Practices](#14-notes-and-best-practices)
  - [2. Convolutional Neural Networks (CNNs)](#2-convolutional-neural-networks-cnns)
    - [2.1 Common Layers and Parameters](#21-common-layers-and-parameters)
    - [2.2 CIFAR-10 Example](#22-cifar-10-example)
    - [2.3 Transfer Learning](#23-transfer-learning)
    - [2.4 Custom Datasets](#24-custom-datasets)
    - [2.5 Popular CNN Architectures](#25-popular-cnn-architectures)
  - [3. Recurrent Neural Networks (RNNs)](#3-recurrent-neural-networks-rnns)
    - [3.1 Simple RNN](#31-simple-rnn)
    - [3.2 LSTMs](#32-lstms)
      - [Gated Recurrent Units (GRUs)](#gated-recurrent-units-grus)
    - [3.3 Time Series](#33-time-series)
    - [3.4 RNN Architectures](#34-rnn-architectures)
      - [Sequence to Sequence Models: Seq2Seq](#sequence-to-sequence-models-seq2seq)
  - [4. Other Topics](#4-other-topics)
    - [Autoencoders and Functional API](#autoencoders-and-functional-api)
      - [Example 1: Compression of MNIST and Functional API](#example-1-compression-of-mnist-and-functional-api)
      - [Example 2: De-noising MNIST](#example-2-de-noising-mnist)
    - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)

## 1. Introduction: Basics

This section introduces how to build Multi-Layer Perceptrons or fully connected neural networks for **regression** and **classification**. As an example, the tabular [Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes) is used in a binary classification context. The notebook can be found here:

[`05d_LAB_Keras_Intro.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/05d_LAB_Keras_Intro.ipynb)

:warning: **Important note**: Random forests and gradient boosting methods (e.g., XGBoost) often outperform neural networks for tabular data; that's the case also here. Therefore, the following is only a usage example.

Most important and re-usable code blocks:

1. Imports
2. Load and prepare dataset: split + normalize
3. Define model: Sequential + Compile (Optimizer, Loss, Metrics)
4. Train model
5. Evaluate model and Inference
6. Save and Load

```python

#####
## 1. Imports
#####

# Import basic ML libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score

# Import Keras objects for Deep Learning
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

#####
## 2. Load and prepare dataset: split + normalize
#####

# Load in the data set 
names = ["times_pregnant",
         "glucose_tolerance_test",
         "blood_pressure",
         "skin_thickness",
         "insulin", 
         "bmi",
         "pedigree_function",
         "age",
         "has_diabetes"]
diabetes_df = pd.read_csv('diabetes.csv', names=names, header=0)

print(diabetes_df.shape) # (768, 9): very small dataset to do deep learning

# Split and scale
X = diabetes_df.iloc[:, :-1].values
y = diabetes_df["has_diabetes"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11111)

normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

#####
## 3. Define model: Sequential + Compile (Optimizer, Loss, Metrics)
#####

# Define a fully connected model 
# - Input size: 8-dimensional
# - Hidden layers: 2 layers, 12 hidden nodes/each, relu activation
# - Dense layers: we specify number of OUTPUT units; for the first layer we specify the input_shape, too
# - Activation: we can either add as layer add(Activation('sigmoid')) or as parameter of Dense(activation='sigmoid')
# - Without an activation function, the activation is linear, i.e. f(x) = x -> regression
# - Final layer has just one node with a sigmoid activation (standard for binary classification)
model = Sequential()
model.add(Dense(units=12, input_shape=(8,), activation='relu'))
model.add(Dense(units=12, input_shape=(8,), activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Summary, parameters
model.summary()

# Compile: Set the the model with Optimizer, Loss Function and Metrics
model.compile(optimizer=SGD(lr = .003),
                loss="binary_crossentropy", 
                metrics=["accuracy"])
# Other options:
# For a multi-class classification problem
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy']) # BUT: balanced
# For a binary classification problem
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy']) # BUT: balanced
# For a mean squared error regression problem
# model.compile(optimizer='adam',
#               loss='mse')
#
# opt = keras.optimizers.Adam(learning_rate=0.01)
# opt = keras.optimizers.SGD(learning_rate=0.01)
# opt = keras.optimizers.RMSprop(learning_rate=0.01)
# ...
# model.compile(..., optimizer=opt)

#####
## 4. Train model
#####

# Train == Fit
# We pass the data to the fit() function,
# including the validation data
# The fit function returns the run history:
# it contains 'val_loss', 'val_accuracy', 'loss', 'accuracy'
# Always shuffle!
# NOTE: passing teh test split as validation_data is a bad practice
# We should either create a validdation split from the train split
# or use the validation_split parameter!
run_hist = model.fit(X_train_norm,
                         y_train,
                         validation_data=(X_test_norm, y_test),
                         epochs=200,
                         shuffle=True)

# Get used metrics
model.metrics_names

#####
## 5. Evaluate model and Inference
#####

# Two kinds of predictions
# One is a hard decision,
# the other is a probabilitistic score.
y_pred_class_nn_1 = model.predict_classes(X_test_norm) # {0, 1}
y_pred_prob_nn_1 = model.predict(X_test_norm) # [0, 1]

# Print model performance and plot the roc curve
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_1))) # 0.755
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_1))) # 0.798

# Plot ROC
def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} on PIMA diabetes problem'.format(model_name),
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])

plot_roc(y_test, y_pred_prob_nn_1, 'NN')

# Learning curves
run_hist_1.history.keys() # ['val_loss', 'val_accuracy', 'loss', 'accuracy']
fig, ax = plt.subplots()
ax.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()

# Learning curves: Another option
losses = pd.DataFrame(model.history.history)
losses.plot()

# We can further train it!
# That's sensible if curves are descending
# NOTE: passing teh test split as validation_data is a bad practice
# We should either create a validdation split from the train split
# or use the validation_split parameter!
run_hist_ = model.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1000)

# Also: evaluate
# Evaluate the model: Compute the average loss for a new dataset = the test split
model.evaluate(X_test_norm,y_test)

#####
## 6. Save and Load
#####

model.save('my_model.h5')
later_model = load_model('my_model.h5')
later_model.predict(X_test_norm.iloc[101, :])
```

### 1.1 Dropout as Regularization

Dropout is added as a layer when using `Sequential`.

```python
model = Sequential()

model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5)) # probability of each neuron to drop: 0.5

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5)) # probability of each neuron to drop: 0.5

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
```

### 1.2 Early Stopping (as Regularization)

Early stopping can be done by defining it as a callback; we need to pass the validation dataset.

```python
# Early stopping when validation loss stops decreasing
# Early stop is achieved by callbacks
# Arguments:
# - monitor: value to be monitored -> val_loss: loss of validaton data
# - mode: min -> training stops when monitored value stops decreasing
# - patience: number of epochs with no improvement after which training will be stopped
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# NOTE: passing teh test split as validation_data is a bad practice
# We should either create a validdation split from the train split
# or use the validation_split parameter!
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_val, y_val),
          shuffle=True,
          verbose=1,
          callbacks=[early_stop])
```

### 1.3 Tensorboard

Tensorboard is a dashboard that visualizes how the network is trained, eg., the weight values along the epochs are displayed, etc.

Official tutorial: [tensorboard/get_started](https://www.tensorflow.org/tensorboard/get_started).

**Install**: `pip/3 install tensorboard`

**Usage**:

1. We instantiate a TensorBoard callback and pass it to `model.fit()`; the callback logs can save many different data.
2. Then, we launch tensorboard in the terminal: `tensorboard --logdir=path_to_your_logs`.
3. We open the tensorboard dashboard with browser at: [http://localhost:6006/](http://localhost:6006/).

**Arguments to instantiate `TensorBoard` (from the help docstring)**:

- `log_dir`: directory of log files used by TensorBoard
- `histogram_freq`: frequency (in epochs) at which to compute activation and
weight histograms for the layers of the model. If set to 0, histograms
won't be computed. Validation data (or split) must be specified for
histogram visualizations.
- `write_graph`: whether to visualize the graph in TensorBoard. The log file
can become quite large when write_graph is set to True.
write_images: whether to write model weights to visualize as image in
TensorBoard.
- `update_freq`: `'batch'` or `'epoch'` or integer. When using `'batch'`,
writes the losses and metrics to TensorBoard after each batch. The same
applies for `'epoch'`. If using an integer, let's say `1000`, the
callback will write the metrics and losses to TensorBoard every 1000
samples. Note that writing too frequently to TensorBoard can slow down
your training.
- `profile_batch`: Profile the batch to sample compute characteristics. By
default, it will profile the second batch. Set `profile_batch=0` to
disable profiling. Must run in TensorFlow eager mode.
- `embeddings_freq`: frequency (in epochs) at which embedding layers will
be visualized. If set to 0, embeddings won't be visualized

Notes:

- Loss is plotted (smoothed or not) for train & validation splits.
- Images (activation maps?) can be visualized in different stages of the network -- it makes sense for CNNs processing images.
- The graph of the model is visualized.
- Weight (& bias) ranges during epochs visualized.
- Histograms of weights (& biases) during epochs visualized.
- Projector: Really cool data visualization (high-dim data projected).

```python
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

## Early Stopping Callback

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

## Tensorboard Callback

timestamp = datetime.now().strftime("%Y-%m-%d--%H%M")

# WINDOWS: Use "logs\\fit"
# MACOS/LINUX: Use "logs/fit"
# Path where log files are stored needs to be specified
# Log files are necessary for the visualizations done in tensorboard
# Always use `logs/fit` and then what you want (eg, a timestamp) 
log_directory = 'logs/fit/'+ timestamp
# Later, when we launch tensorboard in the Terminal:
# --logdir=logs/fit/<timestamp>

board = TensorBoard(
    log_dir=log_directory,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1)

## Model Definition

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

## Training: We pass th early stop and the (tensor-)board as callbacks

# NOTE: passing teh test split as validation_data is a bad practice
# We should either create a validdation split from the train split
# or use the validation_split parameter!
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test),
          shuffle=True,
          verbose=1,
          callbacks=[early_stop,board]
          )

## Open Tensorboard
# 1) In Terminal, start server
# cd <path/to/our/project>
# tensorboard --logdir=<path/to/your/logs>
# tensorboard --logdir=logs/fit/<timestamp>
# 2) In browser, open dashboard
# http://localhost:6006/
```

### 1.4 Notes and Best Practices

- Neural networks are not the best option for tabular data. Random forests and gradient boosting methods (e.g., XGBoost) often outperform neural networks for tabular data.
- Neural networks require large datasets (many rows).
- Always scale inputs to same region to have also weights in similar regions.
- Use a validation split and plot the learning curves.
- We should also vary the used optimizer, the learning rate, activation functions, etc.
- 1 epoch = all samples traversed.
- Always shuffle samples after one epoch!
- Common **Gradient Descend** approaches:
    - Batch gradient descend: all samples (1 epoch) used to compute the loss and one weight update step.
    - Stochastic GD: one random sample used to compute the loss and one weight weight update step.
    - Mini-batch: a batch of random samples used to compute the loss and one weight weight update step.
- **Regularization** techniques or neural networks:
    - Adding weight penalty to loss function.
    - **Dropout**.
    - Early stopping.
    - Stochastic GD or mini-batch GD regularize the training, too, because we don't fit the dataset perfectly.
- **Loss functions**: [keras/losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses), `tensorflow.keras.losses`
    - Regression:
        - `MeanSquaredError()`
        - `MeanAbsoluteError()`
        - `cosine_similarity()`
        - ...
    - Classification:
        - `BinaryCrossentropy()`
        - Multi-class: `CategoricalCrossentropy()`
        - ...
- Common **Optimizers** (from less to most advanced):
    - Gradient descend with **learning rate**.
    - Gradient descend with **momentum**: use running average of the previous steps; momentum is the factor that scales the influence of all previous steps. Common value: `eta = 0.9`. Often times, the learning rate is chosen as `alpha = 1 - eta`. The effect of using momentum is that we smooth out the steps, as compared to stochastic/gradient descend.
    - Gradient descend with **Nesterov momentum**: momentum alone can overshoot the optimum solution. Nesterov momentum controls that overshooting. The effect is that the steps are even more smooth.
    - **AdaGrad**: Frequently updated weights are updated less. We track the value `G`, sum of previous gradients, which increases every iteration and divide each learning rate with it. Effect: as we get closer to the solution, the learning rate is smaller, so we avoid overshooting.
    - **RMSProp**: Root mean square propagation. Similar to AdaGrad, but more efficient. It tracks `G`, but older gradients have a smaller weight; the effect is that newer gradients have more impact.
    - **Adam**: Momentum and RMSProp combined. We have two parameters to tune, which have these default values:
        - `beta1 = 0.9`
        - `beta2 = 0.999`
    - Which one should we use? Adam and RMSProp are very popular and work very well: they're fast. However, if we have convergence issues, we should try simple optimizers, like stochastic gradient descend.
- Common activation functions:
    - `sigmoid`
    - `tanh`
    - `relu`
    - `tf.keras.layers.LeakyReLU()`
    - `softmax`
    - No activation function = linear, i.e. , `f(x) = x`

## 2. Convolutional Neural Networks (CNNs)

### 2.1 Common Layers and Parameters

**Convolutional Layers** - common settings/parameters:

- **Kernel size**: width and height pixels of the filters; usually square kernels are applied with odd numbers: `3 x 3` (recommended), `5 x 5` (less recommended, because more parameters).
- **Padding**: so that we can use corner/edge pixels as centers for the kernels, we add extra pixels on the edges corner; usually, the added pixels have value 0, i.e., *zero-padding*.
  - If we add no padding, the output activation map will be smaller than the input.
  - To conserve image size: `padding = (F-1)/2` with `F` kernel/filter size.
- **Stride**: movement of the kernel in X & Y directions.
  - Usually same stride is used in X & Y.
  - If `stride > 2` we're dividing the image size by `stride`.
- **Depth**: number of channels; we have input and output channels.
  - Each input image has `n` channels.
  - Each output image/map has `N` channels.
  - We have `N` filters, each with `n` kernels applied to the input image.

Another important layer in CNNs: **Pooling**: Pooling reduces image size by mapping an image patch to a value. Commonly `2 x 2` pooling is done, using as stride the pooling window size (i.e., no overlap). We have different types:

- Max-pooling.
- Average-pooling.

### 2.2 CIFAR-10 Example

This section introduces how to build Convolutional Neural Networks (CNNs) for **classification**. The original notebook can be found here:

[`05e_DEMO_CNN.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/05e_DEMO_CNN.ipynb)

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is used, which consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Check the current [performance results here](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).

The 10 classes are:

<ol start="0">
<li> airplane
<li> automobile
<li> bird
<li> cat
<li> deer
<li> dog
<li> frog
<li> horse
<li> ship
<li> truck
</ol>

In the notebook, the following steps are carried out:

1. Imports
2. Load dataset
3. Prepare dataset: encode & scale
4. Define model
5. Train model
6. Evaluate model

```python

###
# 1. Imports
##

import keras
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

###
# 2. Load dataset
##

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape) # (50000, 32, 32, 3)
print(x_train.shape[0], 'train samples') # 50000 train samples
print(x_test.shape[0], 'test samples') # 10000 test samples

###
# 3. Prepare dataset: encode & scale
##

# Each image is a 32 x 32 x 3 numpy array
x_train[444].shape

# Visualize the images
print(y_train[444]) # [9]
plt.imshow(x_train[444]);

# One-hot encoding in Keras/TF
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train[444] # [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]

# Let's make everything float and scale
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

###
# 4. Define model
##

# Let's build a CNN using Keras' Sequential capabilities

model = Sequential()

# Conv2D has these parameters:
# - filters: the number of filters used (= depth of output)
# - kernel_size: an (x,y) tuple giving the height and width of the kernel
# - strides: an (x,y) tuple giving the stride in each dimension; default and common (1,1)
# - input_shape: required only for the first layer (= image channels)
# - padding: "valid" = no padding, or "same" = zeros evenly;
# When padding="same" and strides=1, the output has the same size as the input
# Otherwise, general formula for the size:
# W_out = (W_in + 2P - F)/S + 1; P: "same" = (F-1)/2 ?
model.add(Conv2D(filters=32,
                   kernel_size=(3,3), # common
                   padding='same', # 
                   strides=(1,1), # common, default value
                   input_shape=x_train.shape[1:]))
# We can specify the activation as a layer (as done here)
# or in the previous layer as a parameter
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# Parameters od MaxPooling2D:
# - pool_size: the (x,y) size of the grid to be pooled; 2x2 (usual) halvens the size
# - strides: assumed to be the pool_size unless otherwise specified
# - padding: assumed "valid" = no padding
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten appears when going from convolutional layers to
# fully connected layers.
model.add(Flatten())
model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes))
model.add(Activation('softmax'))

# Always check number of paramaters!
model.summary()

###
# 5. Train model
##

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(lr=0.0005)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# NOTE: passing teh test split as validation_data is a bad practice
# We should either create a validdation split from the train split
# or use the validation_split parameter!
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=(x_test, y_test),
              shuffle=True)

###
# 6. Evaluate model
##

# Validation loss and Validation accuracy
model.evaluate(x_test, y_test)

# Manual computation of the accuracy
import numpy as np
from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(x_test)
y_true = np.argmax(y_test, axis=1) # undo one-hot encoding
print(accuracy_score(y_true, y_pred))

```

### 2.3 Transfer Learning

This section introduces how to use **transfer learning**. Transfer learning consists in re-using the pre-trained weights of a previous model. Since the first layers contain filters that detect edges and simple shapes, often they can generalize well for new datasets. In general, we can say that a network has two parts: (1) the feature extractor and (2) the classifier/regressor.

Possible transfer learning techniques:

- Only the classifier is trained while the feature extractor weights are frozen: transfer learning
- Additional training of the pre-trained network/backbone (feature extractor): fine tuning
  - We can choose to re-train the entire network using as initialization the pre-trained weights or we can select an amount of layers.
- Which one should we use?
  - The more similar the datasets, the less fine-tuning necessary.

The original example code shown below can be found here:

[`05f_DEMO_Transfer_Learning.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/05f_DEMO_Transfer_Learning.ipynb)

In it, the MNIST dataset is used. First, a model is trained with the digits `0-4`; then, we freeze the *feature layer* weights and apply transfer learning to the model in which only the *classifier layers* are re-trained with the digits `5-9`. The training is faster because we train only the classifier.

Most important steps:

1. Imports
2. Define parameters
3. Define data pre-processing + training function
4. Load dataset + split
5. Train: Digits 5-9
6. Freeze feature layers and re-train with digits 0-4

```python

###
# 1. Imports
###

import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from tensorflow import keras
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras import backend as K

# Used to help some of the timing functions
now = datetime.datetime.now

###
# 2. Define parameters
###

# set some parameters
batch_size = 128
num_classes = 5
epochs = 5

# set some more parameters
img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3

## This just handles some variability in how the input data is loaded
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

###
# 3. Define data pre-processing + training function
###

# To simplify things, write a function to include all the training steps
# As input, function takes a model, training set, test set, and the number of classes
# Inside the model object will be the state about which layers we are freezing and which we are training
def train_model(model, train, test, num_classes):
    # train = (x_train, y_train)
    # test = (x_test, y_test)
    x_train = train[0].reshape((train[0].shape[0],) + input_shape) # (60000, 28, 28, 1)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape) # (60000, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    # Measure time
    t = now()
    # NOTE: passing teh test split as validation_data is a bad practice
    # We should either create a validdation split from the train split
    # or use the validation_split parameter!
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t)) # Measure time

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

###
# 4. Load dataset + split
###

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# create two datasets: one with digits below 5 and the other with 5 and above
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5

x_train.shape # (60000, 28, 28)
y_train.shape # (60000,)
input_shape # (28, 28, 1)

###
# 5. Define model: feature layers + classifier
###

# Define the "feature" layers.  These are the early layers that we expect will "transfer"
# to a new problem.  We will freeze these layers during the fine-tuning process
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

# Define the "classification" layers.  These are the later layers that predict the specific classes from the features
# learned by the feature layers.  This is the part of the model that needs to be re-trained for a new problem
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

# We create our model by combining the two sets of layers as follows
model = Sequential(feature_layers + classification_layers)

# Let's take a look: see "trainable" parameters
model.summary()

###
# 5. Train: Digits 5-9
###

# Now, let's train our model on the digits 5,6,7,8,9
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)


###
# 6. Freeze feature layers and re-train with digits 0-4
###

# Freeze only the feature layers
for l in feature_layers:
    l.trainable = False

model.summary() # We see that the "trainable" parameters are less

train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

```

### 2.4 Custom Datasets

In order to work with custom datasets, as in Pytorch, we need to have the following underlying structure:

```
train/
    class_1/
        file_1.jpg
        file_2.jpg
        ...
    class_2/
    ...
    class_n/
test/
    class_1/
    class_2/
    ...
    class_n/
```

With that file structure, we create an `ImageDataGenerator` object and use it in `model.fit_generator()`. The `ImageDataGenerator` performs **data augmentation**, too!

The following example shows how to:

- Instantiate `ImageDataGenerator` with transformation values.
- Create train and test iterators.
- Define a CNN model.
- Fit the model with generator/iterators.
- Load an image without the generators/iterators and perform inference with it.

Th original code can be found here:

[`08_3_Keras_Custom_Datasets.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/08_3_Keras_Custom_Datasets.ipynb)

```python
import matplotlib.pyplot as plt
import cv2
%matplotlib inline

path = './'
img = cv2.imread(path+'CATS_DOGS/train/CAT/0.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
img.shape # (375, 500, 3) - random sized images, need to be resized
img.max # 255 - need to be scaled

# Data augmentation: for more robust and generalizable trainings
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# We pass max absolute values of ranges for data augmentation
# rotation: degrees by which image can be rotated
# width, height shift: % that the image width/height can be changed
# rescale: normalization = divide by max pixel value
# shear: % by which image can be stretched from the diagonal
# zoom: % of image augmenation
# horizontal flip: flip also images?
# fill mode: when pixels are created/removed (eg., when rotating), which values do we take?
# IMPORTANT NOTE: another option is image_dataset_from_directory
# shown in the next section 2.5
image_gen = ImageDataGenerator(rotation_range=30,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              rescale=1/255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')

# We can apply this augmentation transform image-wise easily!
# Every time we call it, we have a slighthly different transformed image
plt.imshow(image_gen.random_transform(img))

# We define the input shape
# All images are going to be resized to that shape
input_shape = (150, 150, 3)

# We define also a batch size
# Images are going to be delivered in batches
# A standard size is 16
batch_size = 16

train_image_gen = image_gen.flow_from_directory(path+'CATS_DOGS/train',
                                               target_size=input_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

train_image_gen = image_gen.flow_from_directory(path+'CATS_DOGS/train',
                                               target_size=input_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

# We can get class/category names from folder names
train_image_gen.class_indices

# Define model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Conv2D, Dense, MaxPooling2D

model = Sequential()
# Convolution + Max-Pooling 1
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# Convolution + Max-Pooling 2 (once more, because images are quite complex)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# Convolution + Max-Pooling 3 (once more, because images are quite complex)
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu')) # that's a new feature: we can add activaton separately!
model.add(Dropout(0.5)) # Dropout layer: 50% of neurons shut down randomly
model.add(Dense(1)) # Binary: Cat / Dog?
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
model.summary()

# TRAIN
# usually an epoch is a complete pass of all training images
# if we define steps_per_epoch, then,
# epoch: steps_per_epoch x batch_size images, not all of them
# We can also pass the validation/test split here to check/prevent overfitting
# Very low values are used here to get a fast training; if done seriously, use higher commented values
# NOTE: passing teh test split as validation_data is a bad practice
# We should either create a validdation split from the train split
# or use the validation_split parameter!
results = model.fit_generator(train_image_gen,
                    epochs=150, # 150
                    steps_per_epoch=1, # 1000
                    validation_data=test_image_gen,
                    validation_steps=1) # 300

# Inference
from tensorflow.keras.preprocessing import image

# We can load the image and resize it automatically
# BUT: the image is not a numpy array, so it must be converted
img = image.load_img(img, target_size=input_shape[:2])

# We can display it, but it's an image object
# not a numpy array yet
plt.imshow(img)

# We need to convert it to a numpy array manually
img = image.img_to_array(img)

# And we need to give the image the shape (Sample, W, H, C) = (1, W, H, C) = (1, 150, 150, 3)
import numpy as np
img = np.expand_dims(img, axis=0)

img.shape # (1, 150, 150, 3)

# Class 0/1 inferred
result = model.predict_classes(img) # [[1]]

train_image_gen.class_indices # {'CAT': 0, 'DOG': 1}

# Raw probability should be predicted
result = model.predict(img) # [[1.]]
```

### 2.5 Popular CNN Architectures

- LeNet
  - Yann LeCun, 1990
  - First CNN
  - Back & white images; tested with MNIST
  - Three times: Conv 5x5 + Subsampling (pooling); then, 2 fully connected layers. 
- AlexNet
  - It popularized the CNNs.
  - Turning point for modern Deep Learning.
  - 16M parameters.
  - They parallelized the network to train in 2 GPUs.
  - Data augmentation was performed to prevent overfitting and allow generalization.
  - ReLUs were used: huge step at the time.
- VGG
  - It simplified the choice of sizes: it uses only 3x3 kernels and deep networks, which effectively replace larger convolutions.
  - The receptive field of two 3x3 kernels is like the receptive field of one 5x5 kernel, but with less parameters! As we go larger, the effect is bigger.
  - VGG showed that many small kernels are better: deep networks.
- Inception
  - The idea is that we often don't know which kernel size should be better applied; thus, instead of fixing on size, we apply several in parallel for each layer and then we concatenate the results.
  - In order to control the depth for each branch, 1x1 convolutions were introduced.
- ResNet
  - VGG showed the power of deep networks; however, from a point on, as we goo deeper, the performance decays because:
    - Early layers are harder too update
    - Vanishing gradient
  - ResNet proposed learning the residuals; in practice, that means that we add the output from the 2nd previous layer to the current output -- these are teh so called **shortcut connections**. As a result, the delta is learned and the previous signal remains untouched. With that, we alleviate considerably the vanishing gradient issue and the networks can be **very deep**!

All these architectures are available at: [Keras Applications: Pre-trained Architectures](https://keras.io/api/applications/).

We can import the architectures with pre-trained weights and apply transfer learning with them.

```python
import matplotlib.pyplot as plt
import numpy as np
import PIL

import tensorflow as tf
from tensorflow.keras import layers,Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Get the data
# Let's suppose a dataset arranged as explained in the previous section
# with 5 classes
classnames = ['A', 'B', 'C', 'D', 'E']

# Create training and validation image iterators
# NOTE: another option would be ImageDataGenerator, show in previous section 2.4
# NOTE: passing teh test split as validation_data is a bad practice
# We should either create a validdation split from the train split
# or use the validation_split parameter!
img_height,img_width=180,180
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Plot some images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(classnames[labels[i]])
    plt.axis("off")*

#####
## Transfer Learning
#####

# Empty sequential model
resnet_model = Sequential()

# From Keras Applications, we can download many pre-trained models
# If we specify include_top=False, the original input/output layers
# are not imported.
# Note that we can specify our desired the input and output layer sizes!
pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',
                   classes=5,
                   weights='imagenet')
# Freeze layers
for layer in pretrained_model.layers:
        layer.trainable=False

# Add ResNet to empty sequential model
resnet_model.add(pretrained_model)

# Now, add the last layers of our model which map the extracted features
# too the classes - that's the classifier, what's really trained
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(5, activation='softmax'))

resnet_model.summary()

resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

# Train/Fit
history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10)

# Inference
import cv2

image = cv2.imread('./path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized= cv2.resize(image, (img_height, img_width))
image=np.expand_dims(image_resized,axis=0)

pred=resnet_model.predict(image)
output_class=class_names[np.argmax(pred)]
```

## 3. Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNN) deal with sequential data. They capture the context information or the structural properties of the sequence by storing a memory or hidden state, which is passed from the previous step to the next as the elements of the sequence are fed step by step. The less sophisticated layers (`SimpleRNN()`) suffer from the vanishing gradient problem, thus, not very long sequences can be handled. The more sophisticated layers (e.g., `LSTM()`) alleviate that issue.

RNNs can be used for:

- Language modeling: sentiment analysis, predict next word, etc.
- Time series forecasting.
- etc.

### 3.1 Simple RNN

Model of a simple RNN:

![Simple RNN](./pics/SimpleRNN.png)

Note that the output of the cell is double: the output and the memory/previous hidden state. That memory state is passed automatically from step to step.

If we *unroll* it over time it is like having `N` consecutive layers, where `N` is the sequence length:

![Unrolled RNN](./pics/unrolled_rnn.jpg)

Notation equivalence between both pictures:

    x_t = w_i: word/vector at position t/i in sequence
    s_t = s_i: state at position t/i in sequence
    y_t = o_i: output at position t/i in sequence
    W_x = U: core RNN, dense layer applied to input
    W_s = W: core RNN, dense layer applied to previous state
    W_y = V: final dense layer (in Keras, we need to do it manually afterwards)

Dimensions:

    r = dimension or input vector
    s = dimension of hidden state
    t = dimension of output

    U: r x s -> initialized with kernel_initializer
    W: s x s -> initialized with recurrent_initializer
    V: s x t (in Keras, we need to do it manually afterwards)

Notes:

- The weight matrices `U, W, V` are the same across all steps/positions!
- We usually only care about the last output!
- The backpropagation is done *through time*, thus, the vanishing gradient problem becomes more patent. Therefore, we can't work with very long sequences. A solution to that length limitation are **Long Short-Term Memory (LSTM)** cells (see below).
- We usually **pad** and **truncate** sequences to make them of a fixed length.
- Training is performed with vectors and batches, thus, the input has the shape of `(batch_size, seq_len, vector_size)`.
- If we use words, these are converted to integers using a dictionary and then an **embedding layer** is defined. The embedding layer converts the integers to word vectors in an embedding space of a fixed dimension. The embedding layer conversion is learnt during training. If desired, we can train the embedding to transform similar words to similar vectors (e.g., as measured by their cosine similarity).
- **IMPORTANT REMARK**: in Keras, apparently, the mapping `V = W_y` is not done automatically inside the `SimpleRNN`, we need to do it manually with a `Dense()` layer, if desired.

The following example shows how to:

1. Imports
2. Load dataset. process, define parameters
3. Define RNN model
4. Train and evaluate RNN model

Th original code can be found here:

[`05g_DEMO_RNN.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/05g_DEMO_RNN.ipynb)

```python
###
# 1. Imports
##

#from tensorflow import keras
#from tensorflow.keras.preprocessing import sequence
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Embedding
#from tensorflow.keras.layers import SimpleRNN
#from tensorflow.keras.datasets import imdb
#from tensorflow.keras import initializers
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import SimpleRNN
from keras.datasets import imdb
from keras import initializers

###
# 2. Load dataset. process, define parameters
##

max_features = 20000  # This is used in loading the data, picks the most common (max_features) words
maxlen = 30  # maximum length of a sequence - truncate after this
batch_size = 32

# Load in the data.
# The function automatically tokenizes the text into distinct integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
# 25000 train sequences
# 25000 test sequences

# This pads (or truncates) the sequences so that they are of the maximum length
# The length of the sequence is very important:
# - if too short, we might fail to capture context correctly
# - if too long, the memory cannot store all the information
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

x_train[123,:]  #Here's what an example sequence looks like
# array([  219,   141,    35,   221,   956,    54,    13,    16,    11,
#         2714,    61,   322,   423,    12,    38,    76,    59,  1803,
#           72,     8, 10508,    23,     5,   967,    12,    38,    85,
#           62,   358,    99], dtype=int32)

###
# 3. Define RNN model
##

rnn_hidden_dim = 5 # dim of hidden state = dim of the output
word_embedding_dim = 50 # dim of word vectors
model_rnn = Sequential()
# Embedding: This layer takes each integer in the sequence
# and embeds it in a 50-dimensional vector
model_rnn.add(Embedding(input_dim=max_features, # vocabulary size
                        output_dim=word_embedding_dim)) # word vector size
# A SimpleRNN is the recurrent layer model with the mappings U, W, V = W_x, W_s, W_y
# The hidden state is passed automatically after each step in the sequence
# The size of the output is the size of the hidden state
# Usually the last output is taken only
# The kernel (U = W_x) and recurrent (W = W_s) mappings can be controlled
# independently for initialization and regularization
# IMPORTANT REMARK: in Keras, apparently, the mapping V = W_y is not done automatically,
# we need to do it manually with a Dense() layer
model_rnn.add(SimpleRNN(units=rnn_hidden_dim, # output size = hidden state size
                    # U = W_x: input weights: (word_embedding_dim, rnn_hidden_dim) = (50, 5)
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    # W = W_s: hidden state weights: (s, s) = (5, 5)
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu', # also frequent: tanh
                    input_shape=x_train.shape[1:]))

# Sentiment analysis: sentiment score
model_rnn.add(Dense(units=1, activation='sigmoid'))

# Note that most of the parameters come from the embedding layer
model_rnn.summary()

rmsprop = keras.optimizers.RMSprop(lr = .0001)

model_rnn.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

###
# 4. Train and evaluate RNN model
##

# NOTE: passing teh test split as validation_data is a bad practice
# We should either create a validdation split from the train split
# or use the validation_split parameter!
model_rnn.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test))

score, acc = model_rnn.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score) # 0.4531511457252502
print('Test accuracy:', acc) # 0.7853999733924866

```

### 3.2 LSTMs

Simple RNN layers suffer from the vanishing gradient problem, so we cannot handle very long sequences. Schmidhuber published in 1997 the **Long Short-Term Memory** units, which alleviate that issue and are still used nowadays.

The LSTM cells have several gates that decide which information to forget and to remember, and their memory has two parts: short-term memory and long-term memory.

The math is in reality not very complex: we apply several sensible operations to the vectors:

- Linear mappings
- Concatenation
- Element-wise multiplication
- Activation with `tanh` and `sigmoid`
- etc.

However, the key aspects of how LSTMs work are summarized by the following picture/model:

![LSTM Unit](./pics/LSTMs.png)

More information can be found in my DL notes: [deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity)

In practice, we can easily interchange the `SimpleRNN()` and the `LSTM()` layers, because for the user the have a very similar API:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM

### SimpleRNN

n_features = 1 # size of a vector; in this case the vector is a scalar
length = 50 # length of vector sequence

model = Sequential()
model.add(SimpleRNN(units=100, input_shape=(length,n_features)))
model.add(Dense(n_features)) # Unique vector output

model.compile(optimizer='adam',loss='mse')

### LSTM

n_features = 1 # size of a vector; in this case the vector is a scalar
length = 50 # length of vector sequence

model = Sequential()
model.add(LSTM(units=100, input_shape=(length,n_features)))
model.add(Dense(n_features)) # Unique vector output

model.compile(optimizer='adam',loss='mse')

```

#### Gated Recurrent Units (GRUs)

They appeared in 2014. They are a simplification of the LSTM cell which is maybe less accurate but requires less memory and are faster.

We can easily interchange them both; maybe LSTMs are able to learn more complex patterns and GRUs are suited for smaller datasets.

### 3.3 Time Series

Time series can be performed with a *many-to-one* architecture, but several aspects should be taken into account:

- The sequence length must capture the trend and the seasonality, i.e., the low and high frequency components of the series.
- We should apply early stopping.
- Dates need to be converted to `datetime`, probably.
- The class `TimeseriesGenerator` is very helpful to generate sequences from a time series.

The following example is from the notebook

[`19_07_2_RNN_Example_1_Sales.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/19_07_2_RNN_Example_1_Sales.ipynb)

which originally comes from J.M. Portilla's course on Tensforlow 2. The example works on a time series downloaded from the FRED website: [Retail Sales: Clothing and Clothing Accessory Stores (Not Seasonally Adjusted)](https://fred.stlouisfed.org/series/RSCCASN). It consists of 334 entries of monthly date-sales pairs. It is a very simple dataset, but the example shows how to deal with a time series in general.

These steps are carried out:

1. Imports
2. Load dataset and prepare
3. Generator
4. Define the model
5. Train the model
6. Forecasting: Test Split + Future (New Timestamps)

```python
###
# 1. Imports
###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

###
# 2. Load dataset and prepare
###

# We load the dataset
# We convert the date string to datetime type on the fly
# and be set that column to be the index
# If we have datetime values, we can use
# - parse_dates=True, or
# - infer_datetime_format=True
df = pd.read_csv('./RSCCASN.csv', parse_dates=True, index_col='DATE')
df.shape # (334, 1)

# We change the name of the column so that it's easier to remember
# Note that it contains sales in millions by day
df.columns = ['Sales']

# We can see that a year (12 months or data points) y a cycle or period
# which contains the major trend and a seasonality components.
# Thus, we need to take that time span into consideration for splitting
df.plot(figsize=(16,6))

# Train/Test Split
# Due to the observation above,
# we split in the last 1.5 years = 18 months
test_size = 18
test_ind = len(df) - test_size
train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

# Scaling
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

###
# 3. Generator
###

# Time series generator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# The length of the input series must be smaller than the length of the test split
# if we do early-stopping validation
length = 12
batch_size = 1
generator = TimeseriesGenerator(data=scaled_train,
                                targets=scaled_train,
                                length=length,
                                batch_size=batch_size)

# We check the first (X,y) pair of the generator
X,y = generator[0]
X.shape # (1, 12, 1)
y.shape # (1, 1)

###
# 4. Define the model
###

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

n_features = 1
model = Sequential()
#model.add(SimpleRNN(units=100, input_shape=(length,n_features)))
# We explicitly use the ReLu activation
model.add(LSTM(units=100, activation='relu', input_shape=(length,n_features)))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)
# We need to create a validation generator
# The length is the same as before,
# taking into account that it must be shorter than the length of the validation split
validation_generator = TimeseriesGenerator(scaled_test,
                                          scaled_test,
                                          length=length,
                                          batch_size=1)

###
# 5. Train the model
###

# We train with an early stop callback
model.fit_generator(generator,
                    epochs=20,
                    validation_data=validation_generator,
                    callbacks=[early_stop])

# We get the loss values and plot them
losses = pd.DataFrame(model.history.history)
losses.plot()

###
# 6. Forecasting
###

### 6.1 Test Split

# We forecast one by one all the values in the test split
# For that, the batch previous to the test split is taken
# a prediction done for it, and then,
# the batch is moved in time to contain predicted values
test_predictions = []
current_batch = scaled_train[-length:].reshape((1,length,n_features))
for i in range(len(test)):
    predicted = model.predict(current_batch)[0]
    test_predictions.append(predicted)
    current_batch = np.append(current_batch[:,1:,:],[[predicted]],axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
test['LSTM Predictions'] = true_predictions

test.plot(figsize=(10,5))

# Compute the RMSE
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(test['Sales'],test['LSTM Predictions']))

### 6.2 Future: New Timestamps

scaled_full_data = scaler.transform(df)

forecast = []
periods_into_future = 24
current_batch = scaled_full_data[-length:].reshape((1,length,n_features))
for i in range(periods_into_future):
    predicted = model.predict(current_batch)[0]
    forecast.append(predicted)
    current_batch = np.append(current_batch[:,1:,:],[[predicted]],axis=1)

# We inverse the scaling
forecast = scaler.inverse_transform(forecast)

# Pick last date: 2019-10-01
df.tail()

# Since we finish on 2019-10-01 with stapsize of 1 month in our full dataset
# we take the next day as start day and the frequency tring 'MS' for monthly data
# More on freq strings: google("pandas frequency strings")
# https://stackoverflow.com/questions/35339139/what-values-are-valid-in-pandas-freq-tags
forecast_index = pd.date_range(start='2019-11-01',
                               periods=periods_into_future,
                               freq='MS')

# Create a dataframe
forecast_df = pd.DataFrame(forecast,index=forecast_index)
forecast_df.columns = ['Sales']

# Plot
ax = df.plot(figsize=(16,6))
forecast_df.plot(ax=ax)
# We can zooom in if desired
plt.xlim('2018-01-01','2021-02-01')
```

### 3.4 RNN Architectures

We can arrange RNN cells in different ways:

- **Sequence-to-vector** (aka. *many to one*): we pass a sequence and expect an element. For example, we can use that architecture to generate text or forecast the next value. The *one* element is in general a vector, but if that vector is of size `1`, it's a scalar; e.g., in price forecasting we predict one price value.
- **Vector-to-sequence** (aka. *one to many*): for instance, given a word, predict the next 5. That *one* vector can be a scalar, too, as before.
- **Sequence-to-sequence** (aka. *many to many*): we pass a sequence and expect a sequence. For example, we could train a chatbot with Q-A sequences or perform machine translation.

#### Sequence to Sequence Models: Seq2Seq

Sequence to sequence models can be used for instance in machine translation. They have an **encoder-decoder** architecture:

- The encoder can be an RNN: we pass a sequence of words (sentence) which finishes with special token `<EOS>` (i.e., *end of sentence*).
- We take the last hidden state: it contains information of the complete sequence we introduced.
- We pass to the decoder the last hidden state of the encoder as the initial state and the initial special token `<SOS>` (i.e., *start of sentence*).
- The decoder starts outputting a sequence of words/tokens step by step and we collect them.
- We input the last token in the next step.
- The sequence ends when the decoder outputs `<EOS>`.

![Sequence to sequence models](./pics/seq2seq.jpg)

However, the explained approach can be improved:

- We can use **beam search**: the decoder outputs in each step probabilities for all possible words; thus, we can consider several branches of possible sentences, instead of taking one word at a time (aka. *greedy search#*). Since the selected word conditions the next output, it is important which word we select. Beam search consists in performing a more complex selection that considers several options, which lead to several sentences.
- Instead of passing the final hidden state from the encoder, we can pass all the intermediate hidden states and apply **attention**. Attention consists in inputing the hidden state which is most similar to the last output. That is achieved, e.g., by measuring the cosine similarity. This is useful in language translation, since the word order in different languages is not the same.

## 4. Other Topics

In the following, some short code snippets and links to other architectures are provided. These architectures build up in the previously explained ones: MLP, CNN, RNN. This guide should be a catalogue of building DL blocks with Keras, not a guide on deep learning.

### Autoencoders and Functional API

PCA can find compressed representations of data, e.g., images. We can use those representations to detect anomalies/defects, reduce noise, remove background, etc.

However, PCA is a **linear** combination of principal elements/components which capture the vectors with maximum variance. In contrast, we might have non-linear relationships between basic components. That can be accomplished with autoencoders.

Autoencoders have these parts:

- Encoder: layers that compress the dimensionality of the input vector to a **compressed** or **latent representation**, also known as **bottleneck**.
- Decoder: layers that upscale the compressed or latent representation to a vector of the dimension as the input.

For training, the loss is defined as the difference between the input and output vectors; then, the gradient of that loss is propagated through the entire network (encoder + decoder).

Applications of autoencoders:

- Compress data, dimensionality reduction
- Noise reduction in data, e.g., images
- Sharpening of images/data
- Identifying key components
- Anomaly detection
- Similarity: we can find similar images by checking their compressed representation vectors, e.g., with cosine similarity.
- Machine translation: we reduce dimensionality and improve the translation process (machine translation has typically a high dimensionality).
- Generation of data, e.g., images. However, typically **variational autoencoders** are used and often **GANs** are superior.
- Neural inpainting: if we remove a part from an image, we can reconstruct it (e.g., remove watermarks).

A type of autoencoders are **Variational autoencoders**, which work the same way as regular autoencoders, except the latent space contains normal distributions instead of scalar values, i.e., the parameters (mean and std. deviation) of distributions are obtained. Then, the decoder samples those distributions and upscales them.

The main application of variational autoencoders is **image generation**. This is the workflow:

- We compress an image to latent vectors: `mu = [m1, m2, ...]` and `sigma = [s1, s2, ...]`.
- We create a latent representation with noise as follows: `x = mu + sigma*noise`; `noise = N(0,1)`
- We feed `x` to the decoder, which upscales it to be an image.

The loss function has two terms which are summed:

- Reconstruction loss: Pixel-wise difference between input and output vectors/images. Binary crossentropy can be used (as done with regular autoencoders).
- Penalty for generating `mu` and `sigma` vectors which are different from `0` and `1`, i.e., we penalize deviations form the **standard distribution**; the  **Kullback-Leibler (KL)-divergence** formula is used.

#### Example 1: Compression of MNIST and Functional API

The notebook

[`05h_LAB_Autoencoders.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/05h_LAB_Autoencoders.ipynb)

shows shows how the following topics/concepts are implemented:

- Compression/Reconstruction performance of PCA applied on MNIST. Reconstruction performance is measured comparing original and reconstructed images with MSE.
- Compression/Reconstruction efficacy of Autoencoders applied on MNIST: Number of layers and units are changed to compare different models.
- Compression/Reconstruction efficacy of Variational Autoencoders applied on MNIST: Number of layers and units are changed to compare different models.
- The decoder of the Variational Autoencoder is used to generate new MNIST images: the two sigma values are varied and images are reconstructed.

One very interesting thing about the noteboook is that we learn how to use the **functional API** from Keras. That API goes beyond the `Sequential()` models; we can define layers as functions and specify operations with them (e.g., addition, concatenation), as required in more complex models (e.g., ResNets).

Example of how to build an Autoencoder and a Variational Autoencoder using the functional API:

```python

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.datasets import mnist

from sklearn.preprocessing import MinMaxScaler

###
# Load and prepare dataset
###

# 60k in train, 10k in test
(x_train, y_train), (x_test, y_test) = mnist.load_data();
# Convert to float and scale to [0,1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# Reshape to 1D, i.e., rows of pixels
x_train_flat = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_flat = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train_flat.shape) # (60000, 784)
print(x_test_flat.shape) # (10000, 784)

# Pixel values scaled again?
s = MinMaxScaler().fit(x_train_flat)
x_train_scaled = s.transform(x_train_flat)
x_test_scaled = s.transform(x_test_flat)

def mse_reconstruction(true, reconstructed):
    return np.sum(np.power(true - reconstructed, 2) / true.shape[1])

###
# Autoencoder
###

# Here the functional API from keras is introduced
# which helps building more complex architectures, e.g., where
# layers outputs are summed or concatenated (e.g., ResNets), going
# beyond the Sequential() API.
# However, for such simple models as the current,
# we could use Sequential().
# In the functional API:
# - We need to define the Input vector
# - We need to define all layers as if they are functions with their outputs
# - We collect first Input instance and last output in Model
#
# We need to define 3 models altogether:
# - full_model
# - encoder
# - decoder

ENCODING_DIM = 64
HIDDEN_DIM = 256

## Encoder model
inputs = Input(shape=(784,)) # input vector
# Functional API: layers are like functions that take arguments
encoder_hidden = Dense(HIDDEN_DIM, activation="relu")(inputs)
encoded = Dense(ENCODING_DIM, activation="sigmoid")(encoder_hidden)
# Model: we collect first Input instance and last output in Model
encoder_model = Model(inputs, encoded, name='encoder')

## Decoder model
encoded_inputs = Input(shape=(ENCODING_DIM,), name='encoding')
decoder_hidden = Dense(HIDDEN_DIM, activation="relu")(encoded_inputs)
reconstruction = Dense(784, activation="sigmoid")(decoder_hidden)
decoder_model = Model(encoded_inputs, reconstruction, name='decoder')

## Full model as the combination of the two
outputs = decoder_model(encoder_model(inputs))
full_model = Model(inputs, outputs, name='full_ae')

full_model.summary()

# We use the BINARY cross-entropy loss
# because we want to know whether the input and output images are equivalent
full_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

## Train
history = full_model.fit(x_train_scaled,
                         x_train_scaled,
                         shuffle=True,
                         epochs=2,
                         batch_size=32)

## Evaluate
decoded_images = full_model.predict(x_test_scaled)
mse_reconstruction(x_test_flat, decoded_images)

###
# Variational Autoencoder
###

def sampling(args):
    """
    Transforms parameters defining the latent space into a normal distribution.
    """
    # Need to unpack arguments like this because of the way the Keras "Lambda" function works.
    mu, log_sigma = args
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=tf.shape(mu))
    sigma = K.exp(log_sigma)
    #return mu + K.exp(0.5 * sigma) * epsilon
    return mu + sigma*epsilon

hidden_dim = 256
batch_size = 128
# this is the dimension of each of the vectors representing the two parameters
# that will get transformed into a normal distribution
latent_dim = 2 
epochs = 1

## VAE model = encoder + decoder

## Encoder model
inputs = Input(shape=(784, ), name='encoder_input')
x = Dense(hidden_dim, activation='relu')(inputs)
# We pass x to two layers in parallel
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x) # 2D vectors

# We can pass to Lambda our own created function
z = Lambda(sampling, name='z')([z_mean, z_log_var]) # args unpacked in sampling()
# z is now one n dimensional vector representing the inputs
# We'll have the encoder_model output z_mean, z_log_var, and z
# so we can plot the images as a function of these later
encoder_model = Model(inputs, [z_mean, z_log_var, z], name='encoder')

## Decoder model
latent_inputs = Input(shape=(latent_dim,),)
x = Dense(hidden_dim, activation='relu')(latent_inputs)
outputs = Dense(784, activation='sigmoid')(x)
decoder_model = Model(latent_inputs, outputs, name='decoder')

## Instantiate VAE model
outputs = decoder_model(encoder_model(inputs)[2]) # we take only z!
vae_model = Model(inputs, outputs, name='vae_mlp')

## Examine layers

for i, layer in enumerate(vae_model.layers):
    print("Layer", i+1)
    print("Name", layer.name)
    print("Input shape", layer.input_shape)
    print("Output shape", layer.output_shape)
    if not layer.weights:
        print("No weights for this layer")
        continue
    for i, weight in enumerate(layer.weights):
        print("Weights", i+1)
        print("Name", weight.name)
        print("Weights shape:", weight.shape.as_list())

## Loss function

# Reconstruction
reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= 784

# KL-Divergence
kl_loss = 0.5 * (K.exp(z_log_var) - (1 + z_log_var) + K.square(z_mean))
kl_loss = K.sum(kl_loss, axis=-1)
total_vae_loss = K.mean(reconstruction_loss + kl_loss)

# We can pass our custom loss function with add_loss()
vae_model.add_loss(total_vae_loss)

vae_model.compile(optimizer='rmsprop',
                  metrics=['accuracy'])
    
vae_model.summary()

## Train
vae_model.fit(x_train_scaled,
        epochs=epochs,
        batch_size=batch_size)

## Evaluate: Generate reconstructed images and compare to original ones
decoded_images = vae_model.predict(x_test_scaled)
mse_reconstruction(x_test_scaled, decoded_images)

```

#### Example 2: De-noising MNIST

Another nice example on autoencoders is given in the notebook:

[`19_09_2_Keras_Autoencoders_Image_Denoising_MNIST.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/19_09_2_Keras_Autoencoders_Image_Denoising_MNIST.ipynb)

In it, an autoencoder is built to de-noise MNIST images. The notebook comes originally from J.M. Portilla's course on Tensorflow 2.

### Generative Adversarial Networks (GANs)

Two notebooks in which a GAN and a DCGAN are implemented on the MNIST dataset:

- [`19_10_1_Keras_GANs_Intro_MNIST.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/19_10_1_Keras_GANs_Intro_MNIST.ipynb)
- [`19_10_2_Keras_GANs_DCGAN_MNIST.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/05_Deep_Learning/lab/19_10_2_Keras_GANs_DCGAN_MNIST.ipynb)

The notebooks come originally from J.M. Portilla's course on Tensorflow 2.
