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
run_hist = model.fit(X_train_norm,
                         y_train,
                         validation_data=(X_test_norm, y_test),
                         epochs=200)

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
          verbose=1,
          callbacks=[early_stop]
          )
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
- **Loss functions**:
    - MSE
    - 
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

## 2. Convolutional Neural Networks (CNNs)

## 3. Recurrent Neural Networks (RNNs)

## 4. Autoencoders

## 5. Generative Adversarial Networks (GANs)