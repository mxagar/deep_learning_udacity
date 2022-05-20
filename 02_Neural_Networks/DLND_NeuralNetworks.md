# Udacity Deep Learning Nanodegree: Introduction

These are my personal notes taken while following the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

The nanodegree is composed of six modules:

1. Introduction to Deep Learning
2. Neural Networks
3. Convolutonal Neural Networks (CNN)
4. Recurrent Neural Networks (RNN)
5. Generative Adversarial Networks (GAN)
6. Deploying a Model

Each module has a folder with its respective notes. This folder is the one of the **second module**: Neural Networks.

Additionally, note that:
- I made many hand-written nortes, which I will scan and push to this repostory.
- I forked the Udacity repository for the exercisesl [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch); all the material and  notebooks are there.

## Overview of Contents

2. Neural Networks
	- Lesson 1: Introduction to Neural Networks
		- List of concepts
		- Notebook: `GradientsDescend.ipynb`	
		- Notebook: `StudentAdmissions.ipynb`
	- Lesson 2: Implementing Gradient Descend
	- Lesson 3: Training Neural Networks
	- Lesson 4: GPU Workspaces Demo
	- Lesson 5: Sentiment Analysis
	- Project: Predicting Bike Sharing Patterns


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
	- That way, the maximum likelihood is the probability of all the classes, and the cross-entropy is the error. The higher the probability, the lower the error!
	- Another way of interpreting the cross-entropy is the distance error from a discrete vector to our continuous probabilities: `CE([1,1,0],[0.8,0.7,0.1]) = 0.69`
- Error Function
- Gradient Descend
- Comparison: Perceptron Algorithm vs Gradient descend
- Nonlinear Models: Multi-layer Perceptrons = Neural Networks
- Feedforward
- Backpropagation

Interesting Jupyter Notebooks: [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch)

- `intro-neural-networks/GradientsDescend.ipynb`: Gradient descend implemented with `numpy` to a 2D pointcloud with two classes. The line defined by the weights is plotted along the time/training epochs. The code is below.
- `intro-neural-networks/StudentAdmissions.ipynb`: Gradient descent implemented with `numpy` to a linear model. The code is below.
	- Dataset: student admission data: 3D data (test result, GPA grades, class rank quantile), converted to one-hot 6D.
	- One-hot encoding is done in pandas using `get_dummies()`: Rank 0-4 -> rank_i 0/1 for i 1-4.
	- Variables are scaled.
	- Train/Test split done with `np.random.choice`

### Notebook: `GradientsDescend.ipynb`

```python
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Some helper functions for plotting and drawing lines
def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)


# Load and visualize the data
data = pd.read_csv('data.csv', header=None)
X = np.array(data[[0,1]])
y = np.array(data[2])
plot_points(X,y)
plt.show()

# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

# Error (log-loss) formula
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

# Gradient descent step
# x: 1 x n_features
# weights: 1 x n_features
# y: 1 x 1
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias) # y_hat
    d_error = y - output
    weights += learnrate * d_error * x # watch out: dE/dw = - ...
    bias += learnrate * d_error
    return weights, bias

# Training Function
def train(features, targets, epochs, learnrate, graph_lines=False):
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        # Go through all samples, for each, update weights & bias with gradient descend
        for x, y in zip(features, targets):
            # That is not completely correct?
            # We should compute the complete batch gradient
            # and then update the weights?!
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        #targets.shape
        #out.shape
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            
            # Converting the output (float) to boolean as it is a binary classification
            # e.g. 0.95 --> True (= 1), 0.31 --> False (= 0)
            predictions = out > 0.5
            
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
    
    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()

# Apply all            
np.random.seed(44)
epochs = 100
learnrate = 0.01
train(X, y, epochs, learnrate, True)

```

### Notebook: `StudentAdmissions.ipynb`

Relevant pieces of code:

```python
# Dummy variables of rank: 1, 2, 3, 4 -> [1/0, 1/0, 1/0, 1/0]
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
one_hot_data = one_hot_data.drop('rank', axis=1)

# Manual Train/Test Split
sample = np.random.choice(processed_data.index,
						size=int(len(processed_data)*0.9),
						replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)
```

## Lesson 2: Implementing Gradient Descend

This lesson introduces very few new concepts; instead, the math behind the gradient descend is reviewed and implemented in code. Note that until now the cross-entropy loss has been used (classification). In this lesson, the *squared sums* are introduced, more suited for regression problems.

**Look at the handwritten nodes**. In them, backpropagation is derived. In the following, code examples are provided.

### Gradient Descend in Numpy: Basic Idea

`w_k <- w_k + dw_k`  
`dw_k = learning_rate * error_term * x_k`

![Gradient Descend Formulas](./pics/gradient_descend_formulas.png)

### One Perceptron, One Data-Point

```python
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

# Calculate the node's linear combination of inputs and weights
h = np.dot(w,x)

# Calculate output of neural network
nn_output = sigmoid(h)

# Calculate error of neural network
error = (y-nn_output)

# Calculate the error term
error_term = (y-nn_output)*sigmoid_prime(h)

# Calculate change in weights
del_w = learnrate*error_term*x
```

### One Perceptron, Several Data Points

The dataset used is the one from `StudentAdmissions.ipynb`:

```
admit,gre,gpa,rank
0,380,3.61,3
1,660,3.67,3
...
```

with:

- `admit`: admission or not
- `gre`: test score
- `gpa`: grade point average
- `rank`: rank quantile in class: `1, 2, 3, 4`

The dataset can be downloaded from here:

[http://www.ats.ucla.edu/stat/data/binary.csv)](http://www.ats.ucla.edu/stat/data/binary.csv))

The rank is encoded as a one-hot variable and the test result and the score are scaled. The goal is to build a model that predicts admission.

The basic gradient descend algorithm for one perceptron is:

![Gradient Descend Algorithm](./pics/gradient_descend_algorithm.png)

with

- `E = MSE = 0.5 * m * sum(j = 1:m datapoints; (y(j) - y_pred(j))^2)`
- `f(h) = sigmoid(h) = 1/(1 + exp(-h))`
- `f'(h) = f(h)(1 - f(h))`, if `f` is the `sigmoid`
- `h = sum(i = 1:n features; w_i * x_i)`

This algorithm is implemented below.

```python
import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

### -- Data Preparation

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

### -- Gradient Descend

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s*(1-s)
    
# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Important: Initialize weights
# We need to break symmetry and allow for weight divergence
# Typical random values: N(0, 1/sqrt(num_features))
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 500
learnrate = 0.01

for e in range(epochs):
	# Weight change
    del_w = np.zeros(weights.shape)
    # Loop through all records, x is the input, y is the target
    for x, y in zip(features.values, targets):
        # Calculate the output: y_pred
        h = np.dot(weights, x)
        output = sigmoid(h)

        # Calculate the error
        error = y-output

        # Calculate the error term: delta
        error_term = error*sigmoid_prime(x)

        # Calculate the change in weights for this sample
        # and add it to the total weight change
        del_w += error_term*x

    # Update weights using the learning rate
    # and the average change in weights
    # Note that the weight change is `w_new = w_old - dE/dw`;
    # however, the error term `delta` contains already the `-` sign,
    # thus: `w_new = w_old + Dw; Dw = (lr/m)*sum(delta_j)`
    weights += (learnrate/n_records)*del_w

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

```

Some notes:

- Dummy variables for rank make sense because `rank = 2` is not `2x` `rank = 1`.
- Scaling is fundamental also because the sigmoid function squashes large and small values: the gradient becomes `0`.
- Taking the Mean Square Error (MSE) instead of the Sum of Square Error (SSE), makes the error and the learning rate to be in a known region independently of the size of the dataset.
- Typical learning rates are in `[0.001, 0.01]`
- Note that the weight change is `w_new = w_old - dE/dw`; however, the error term `delta` contains already the `-` sign, thus: `w_new = w_old + Dw; Dw = (lr/m)*sum(delta_j)`.
- The final accuracy is very low: 0.5

### Multiple Perceptrons (MLP), Several Data Points, One Forward Pass

See handwritten notes for this case, since the matrix derivation and the notation used are important.

A Multilayer Perceptron (MLP) should increase the accuracy.

The following architecture is used:

- Input layer with 3 units
- Hidden layer with 2 units
- Output layer with 1 unit

![MLP Architecture](./pics/mlp_architecture.png)

The unit `j` of the hidden layer is computed as follows:

`h_j = sum(i; x_i * w_ij) = x_1 * w_1j + x_2 * w_2j + x_3 * w_3j`  
`out_j = sigmoid(h_j)`

And everything is packed in matrices, as follows:

`[h_1, h_2] = [x_1, x_2, x3] x [[w_11, w_21, w_31]^T, [w_12, w_22, w_32]^T]`

Or we can use the transpose version, with column vectors: `h (2x1) = W (2x3) x X (3x1)`.

Note on transposing `numpy` arrays:

```python
# 1D: 1x3; shape = (3,), row vector
x = np.array([ 0.49671415, -0.1382643 ,  0.64768854])
# The transpose of 1D arrays is still a row vector!
# However, this does not happen with matrices
x.T # shape = (3,)
# To obtain a column vector from a row vector:
# 3x1
x[:, None] # shape = (3,1)
# Also, to obtain a column vector
# we can use ndim
# but the shape is different!
np.array(x, ndmin=2).T # (1,3)
```

Example of **one forward pass**:

```python
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Some fake data: X, 1 x N_input or (N_input,)
X = np.random.randn(4)

# N_input x N_hidden
weights_input_to_hidden = np.random.normal(
	0,
	scale=0.1,
	size=(N_input, N_hidden))
# N_hidden x N_output
weights_hidden_to_output = np.random.normal(
	0,
	scale=0.1,
	size=(N_hidden, N_output))

# A forward pass through the network
# X: 4 -> 4x1
# w_in: 4x3 -> 3x4
# w_in x X: 3x1
# w_hidden: 3x2 -> 2x3
hidden_layer_in = X[:,None] # (4,1)
hidden_layer_out = sigmoid(np.matmul(weights_input_to_hidden.T, X)) # (3,4)x(4,1)

print('Hidden-layer Output:')
print(hidden_layer_out) # (3,1)

output_layer_in = hidden_layer_out
output_layer_out = sigmoid(
	np.matmul(
		weights_hidden_to_output.T,
		output_layer_in)) # (2,3)x(3,1)

print('Output-layer Output:')
print(output_layer_out) # (2,1)
```

### Backpropagation: One Backward Pass

See my handwritten notes and my notes after following the Machine Learning course by Andrew Ng at Coursera:

[Neural Networks](https://github.com/mxagar/machine_learning_coursera)/`03_NeuralNetworks/ML_NeuralNetworks.md`

The basic idea is that we propagate backwards the error in order to compute the error term `delta` of each unit. Note that the error terms are related to the neuron units; they are used to compute the weight updates.


### Implementation of Backpropagation

This section computes an MLP with 3 input units, 1 hidden layer with 2 units and an output layer with one unit. The dataset is the same as in previous examples.

Have a look at my handwritten notes; the notation is slightly different to the one used by Udacity, but I think it is easier to understand:

`./NeuralNetworks_Backpropagation_Training.pdf`

Also, note that my handwritten notes consider the bias nodes; here there is no bias. That makes the coding more simple.

For a nice implementation of the backpropagation algorithm, have a look at

[Neural Networks](https://github.com/mxagar/machine_learning_coursera)/`03_NeuralNetworks/ML_NeuralNetworks.md`

Implementation code.

```python
import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

### -- Data Preparation

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

### -- MLP Training with Backpropagation

np.random.seed(21)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):

        ## Forward pass ##

        # a(1) = x
        # z(2) = hidden_input
        hidden_input = np.dot(x, weights_input_hidden)
        # a(2) = hidden_output
        hidden_output = sigmoid(hidden_input)
        # a(3) = output
        output = sigmoid(np.dot(weights_hidden_output, hidden_output))

        ## Backward pass ##

        # Calculate the network's prediction error
        error = y-output

        # Calculate error term for the output unit
        # delta(3)
        output_error_term = error*output*(1-output)

        # Propagate errors to hidden layer

        # Calculate the hidden layer's contribution to the error
        # delta(2) <- delta(3) * w(2)
        hidden_error = np.dot(output_error_term, weights_hidden_output)
        
        # Calculate the error term for the hidden layer
        # delta(2) <- delta(2) * f'(z(2)) = delta(2) * f(z(2))*(1-f(z(2))
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)
        
        # Update the change in weights
        # W(2): DW(2) = delta(3)*a(2)
        # W(1): DW(1) = delta(2)*a(1), a(1) = x
        del_w_hidden_output += output_error_term*hidden_output
        del_w_input_hidden += hidden_error_term*x[:,None]

    # Update weights  (don't forget to division by n_records or number of samples)
    weights_input_hidden += (learnrate/n_records)*del_w_input_hidden
    weights_hidden_output += (learnrate/n_records)*del_w_hidden_output

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

```

### Interesting Links

- [Why Momentum Really Works](https://distill.pub/2017/momentum/): Momentum is a possible solution to avoiding local minima.
- [Yes, you should understand backprop; by Andrej Karpathy](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b#.vt3ax2kg9)
- [Lecture on Backpropagation; by Andrej Karpathy](https://www.youtube.com/watch?v=59Hbtz7XgjM).
- My notes on backprop from the Andrew Ng course: [Neural Networks](https://github.com/mxagar/machine_learning_coursera)/`03_NeuralNetworks/ML_NeuralNetworks.md`


## Lesson 3: Training Neural Networks

Same videos (thus, also concepts) as in the Computer Vision Nanodegree (CVND) are treated. Have a look at here:

[computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity)

The goal of the section is to explain how to improve training. In the following, I summarize the most important concepts. For more information, see the handwritten notes in the CVND related to these topics (pages 20-24).

- Complexity of Architectures: select medium complex and prevent overfitting
- Training and Test Splits: the test split tells us how generalizable our model is.
- Overfitting & Underfitting
	- Underfitting: error due to bias; too simplistic model
	- Overfitting: error due to variance; we learn the noise; too complex model
- Methods to prevent overfitting
	- Early Stopping: we track the error metric of both the training and test splits during training epochs - when the error of the test splits starts increasing, we stop.
	- Regularization (L1 & L2): we penalize the weights so that they become smaller; smaller weights have lead to smoother sigmoids, which are more uncertain models, thus, with less curvy regions.
		- L1: sum absolute weights: small weights become 0, good for feature selection
		- L2: sum of squared weights: more homogeneous and small weights 
	- Dropout: we shut down weight update with a probability; thus, we prevent large weights to dominate the training
- Random Restart to avoid falling in Local Minima
- Other activation functions that avoid the vanishing gradient issue: `tanh`, `relu`
	- This is motivated by the small tangent of the sigmoid as we input larger magnitudes
	- Note that we can have a regression model if we leave the `relu` activation at the end 
- Stochastic batch gradient descend: optimization step for each each (mini) batch, not the whole dataset
	- I think originally batch referred to the whole dataset. Here, it is meant mini-batch; however, it has become so popular that "mini" is dropped.
	- The (mini) batches are selected randomly; hence, "stochastic"
- Learning rate: select a small one, decrease as epochs increase
- Momentum (`beta`), to avoid local minima
	- `beta in [0,1)`
	- The effect of using `beta` is that we avoid local minima
	- When the weight update equation is developed with the momentum, it is as if we would consider the previous gradient vectors, multiplied by `beta` powered to the number of epochs/optimization steps before the current one

## Lesson 4: GPU Workspaces Demo

GPU workspace sessions are available: connections from my browser to a remote server.

Each student has limited number of GPU hours allocated.

Only 3Gb data can be stored in workspace home.

Enable/Disable GPU to use it. **`DISABLE` it actively to avoid runnig out of GPU hours!**. Always save before switching to GPU or back.

Workspaces automatically disconnected after 30 mins of inactivity. If we need to, eg., train, longer, use the `workspace_utils.py` utility:

```python
from workspace_utils import active_session

with active_session():
    # do long-running work here
```

The file `workspace_utils.py` shoud be in th workspace, but I downloaded a copy to `./lab/`.

Submitting a project: either

- click on "Submit Project" on Notebook workspace, if available
- or download all files and submit them in classroom

Terminals: available in Jupyter workspace: `New > Terminal`

- we can install things! or use workspace terminal as normally is done;
- toggle Jupyter logo to go back to notebook / workspace viewer.

Menu button:

- Reset data: workspace is deleted and a new ne created; if we dont do that, our workspace data should be saved between different sessions
- Download a copy of our data before doing that!
- Actually, usually you dont need to do that...

## Lesson 5: Sentiment Analysis

Sentiment Analysis with Andrew Trask: NLP PhD Student at Oxford, author of Grokking Deep Learning.

This section is divided in 6 mini-projects. There are 4 files in the repository:

[deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch)`/sentiment-analysis-network`:

- `Sentiment_Classification_Projects.ipynb`: the mini-projects are implemented here
- `Sentiment_Classification_Solutions.ipynb`: solutions
- `reviews.txt`: 25 thousand movie reviews
- `labels.txt`: positive/negative sentiment labels for the reviews

## Project: Predicting Bike Sharing Patterns (Lesson 6)


## Lesson 7: Deep Learning with Pytorch



