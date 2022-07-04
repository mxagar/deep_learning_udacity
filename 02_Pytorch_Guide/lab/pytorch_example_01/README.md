# Transfer Learning with Pytorch

This project contains a transfer learning example of Pytorch, in which the [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) CNN is used to classify a custom dataset of dogs and cats.

The project is a good example of all the steps that need to be carried out when we want to apply DL classification to a custom dataset; these steps are covered:

- Dataset preparation: own annotated files are re-organized into train/test split and class folder (a reduced subset of dogs-vs-cats from Kaggle).
- The DenseNet-121 network is loaded and a classifier is appended to it to detect 2 classes: Cat and Dog.
- Training function is defined and executed with validation accuracy check.
- Training evolution metrics are visualized.
- Functions for saving and load the model are shown.
- A custom `visualize_classify()` function is defined for manal inferences.
- Inference is done in bulk with `infer_images()`: a new set of images is loaded and prepared, inferred and the results are saved to a CSV.

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

- `organize_dataset.py`: given a folder with images and an XLSX file with their annotations, this script (1) separates the image files into class folders and then (2) performs a train/test split with disjoint subsets of images located in separate `train/` and `test` folders.
- `application_notebook.ipynb`: notebook from which all the steps mentioned above are executed: dataset preparation, network definition, training, saving/loading of the model and inference.
- `inference_results.csv`: results of the inference performed in the example.
- `densenet121_trained_cat_dog.pth`: weights and hyperparameters of the saved model.

## Authorship

Mikel Sagardia, 2021.  
No guarantees.

You are free to copy and re-use my code; please, reference my authorship and, especially, the sources I have used.
