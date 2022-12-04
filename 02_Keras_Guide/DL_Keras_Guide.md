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
    - [1.3 Best Practices and Insights](#13-best-practices-and-insights)
    - [1.4 Tensorboard](#14-tensorboard)

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

### 1.3 Best Practices and Insights

- Use a validation split and to plot the learning curves.
- We should also vary the used optimizer, the learning rate, activation functions, etc.
- Neural networks are not the best option for tabular data. Random forests and gradient boosting methods (e.g., XGBoost) often outperform neural networks for tabular data.
- Neural networks require large datasets (many rows).

### 1.4 Tensorboard

