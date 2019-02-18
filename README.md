# Deep-Learning-Apply-various-architectures-on-the-MNIST-dataset.
In this tutorial, we will apply a bunch of various Neural Network Architectures on the MNIST dataset and see how each of them behaves with respect to one another.

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.The database is also widely used for training and testing in the field of machine learning.It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.

The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset. There have been a number of scientific papers on attempts to achieve the lowest error rate; one paper, using a hierarchical system of convolutional neural networks, manages to get an error rate on the MNIST database of 0.23%. The original creators of the database keep a list of some of the methods tested on it. In their original paper, they use a support vector machine to get an error rate of 0.8%. An extended dataset similar to MNIST called EMNIST has been published in 2017, which contains 240,000 training images, and 40,000 testing images of handwritten digits and characters.

Neural Nets:

a. Load the MNIST dataset using Keras.
b. Perform data visualization using PCA and t-SNE
c. Plot the train and validation loss for each architecture and choose the number of epochs for which the model doesn't overfit/underfit.
d. Try the following model architectures: 

1. A simple Softmax Classifier on the MNIST dataset + SGD Optimizer
2. MNIST + ReLU Activation + ADAM optimzer + He initialization. Model Architecture: 784-512-128-10
3. MNIST + ReLU Activation + ADAM optimzer + Dropout + Batch Normalization + He initialization. Model Architecture: 784-512-128-10 ===> 2 Hidden Layers
4. MNIST + ReLU Activation + ADAM optimzer + Dropout + Batch Normalization + He initialization. Model Architecture: 784-784-512-10 ===> 2 Hidden Layers
5. MNIST + ReLU Activation + ADAM optimzer + Dropout + Batch Normalization + He initialization. Model Architecture: 784-512-364-128-10 ===> 3 Hidden Layers
6. MNIST + ReLU Activation + ADAM optimzer + Dropout + Batch Normalization + He initialization. Model Architecture: 784-364-512-256-128-64-10 ===> 5 Hidden Layers
7. MNIST + ReLU Activation + ADAM optimzer + Dropout + Batch Normalization + He initialization. Model Architecture: 784-512-512-512-512-10 ===> 4 Hidden Layers
8. MNIST + ReLU Activation + ADAM optimzer + Dropout + Batch Normalization + He initialization. Model Architecture: 784-32-32-32-32-32-32-10 ===> 6 Hidden Layers

CNN on MNIST Dataset.

a. Load the MNIST dataset using Keras.
b. Perform data visualization using PCA and t-SNE
c. Plot the train and validation loss for each architecture and choose the number of epochs for which the model doesn't overfit/underfit.
d. Try the following model architectures: 

1. CNN Architecture 1: 3 CNN Layers each of 3X3 Kernels.
2. CNN Architecture 2: 5 CNN Layers each of 5X5 Kernels.
3. CNN Architecture 3: 7 CNN Layers each of different Kernels.

e. Plot the train as well as the validation loss to get an idea of when a model is overfitting/underfitting.
