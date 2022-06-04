# Pytorch Examples for Deep Learning Applications

List of examples (they usually build up on the previous):

- `pytorch_example_00`: basic but very complete example in which pytorch is introduced; the **whole image classification pipeline** is practiced: model definition (fully connected and CNN), dataset loading and pre-processing (Fashion-MNIST), training with validation, and inference. Functions are modularized and externalized in python scripts.

- `pytorch_example_01`: **transfer learning (DenseNet121, CNN) applied to a custom dataset** in which images need to be classified (dogs and cats); images are first organized into forlders with a script, and then, the classification pipeline is followed: loading/preprocessing, model transfer, traning, and inference. Inferred classes are saved in a CSV.

## List of Applications / Utilities

- Image Classification

    - `pytorch_example_00`: Fashion-MNIST, manual network (FC & CNN), training-validation-inference, save & load.
    - `pytorch_example_01`: Transfer learning with DenseNet-121, custom dataset handling (subset of dogs-vs-cats), training-validation-inference, save & load.

- Image Object Detection

- Image Segmentation
