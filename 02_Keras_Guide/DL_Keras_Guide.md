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
  - [3. Recurrent Neural Networks (RNNs)](#3-recurrent-neural-networks-rnns)
  - [4. Autoencoders](#4-autoencoders)
  - [5. Generative Adversarial Networks (GANs)](#5-generative-adversarial-networks-gans)

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
run_hist = model.fit(X_train_norm,
                         y_train,
                         validation_data=(X_test_norm, y_test),
                         epochs=200,
                         shuffle=True)

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

## 3. Recurrent Neural Networks (RNNs)



## 4. Autoencoders

## 5. Generative Adversarial Networks (GANs)