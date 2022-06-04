# Basic Usage of Pytorch

This project collects the very basic functionalities of Pytorch:

- Basic tensor operarionst aare shown.
- A fully connected network is defined to classify the Fashion-MNIST dataset.
- Network definition is shown with classes and `Sequential`.
- Custom dataset generation is shown.
- Saving and loading networks is shown.
- A CNN is manually defined.
- Convolutional layers and activation maps are accessed and visualized.

In order to use these examples, we need to install Pytorch in out environment (eg., by using Anaconda):

```bash
conda create -n myenv python=3.6
source activate myenv
conda install opencv-python matplotlib numpy pillow jupyter scipy pandas
conda install pytorch torchvision -c pytorch
# I had some issues with numpy and torch
pip uninstall numpy
pip uninstall mkl-service
pip install numpy
pip install mkl-service
```

For a more detailed guide on Deep Learning and its implementation and use with Pytorch, have a look at my complete repository on that topic:

[deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity)

## Files and content

- `application_notebook.ipynb`: main notebook in which the content of the whole example folder is presented in these sections
    - Tensors: `tensors.py`
    - Main Exemplary Application: main example in which pytorch is showcased; `fullyconnected_neuralnetwork.py` and `helper_nn.py` are used to train a model (defined in the notebook) with Fashion-MNIST dataset and infer some images
    - Custom Datasets: details on how to deal with own image datasets are explained
    - Additional: manually defined networks, access to weight values, and layer/filter & activation/feature map visualization
- `tensors.py`: basic tensor usage.
- `helper_nn.py`: utils for model testing and visualization of dataset & inferred values
- `fullyconnected_neuralnetwork.py`: 
    - class `Network` created; it is fully connected (linear), with relu, dropout (optional), and desired hidden layers with their sizes
    - `validation` function for test split validation during training
    - `train` function
    - `save_model` function; `*.pth` file is saved, with architecture information & weight states
    - `load_model` function