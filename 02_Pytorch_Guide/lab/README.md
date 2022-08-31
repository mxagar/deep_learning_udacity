# Pytorch Examples for Deep Learning Applications

List of examples (they usually build up on the previous):

- `pytorch_example_00`: basic but very complete example in which pytorch is introduced; the **whole image classification pipeline** is practiced: model definition (fully connected and CNN), dataset loading and pre-processing (Fashion-MNIST), training with validation, and inference. Functions are modularized and externalized in python scripts.

- `pytorch_example_01`: **transfer learning (DenseNet121, CNN) applied to a custom dataset** in which images need to be classified (dogs and cats); images are first organized into forlders with a script, and then, the classification pipeline is followed: loading/preprocessing, model transfer, traning, and inference. Inferred classes are saved in a CSV.

- `pytorch_example_02`

- `pytorch_inference_pipeline`

## List of Applications / Utilities

- Image Classification

    - `pytorch_example_00`: Fashion-MNIST (28x28x1), manual network (FC MLP & simple CNN), training-validation-inference, layer visualization, save & load.
    - `pytorch_example_01`: Transfer learning with DenseNet-121, custom dataset handling (subset of dogs-vs-cats from Kaggle, RGB), training-validation-inference, save & load.
    - `pytorch_example_02`: CIFAR-10 (32x32x3) with a manually defined CNN and transfer learning with VGG16. The example is similar to `pytorch_example_00`, but with a more complex dataset; additionally, three splits are done to the dataset: train/validation/test.

- Image Object Detection

- Image Segmentation

- Anomaly Detection

- MLOps
    
    - `pytorch_inference_pipeline`