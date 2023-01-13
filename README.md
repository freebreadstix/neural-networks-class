# neural-networks-class

Project submissions for deep learning class, with abstract included. Can provide code privately if requested


## 4: Experimentation with Various Encoder-Decoder CNN-RNN Architectures for Image Captioning
* abstract accidentally not changed, excerpt from introduction *

In this study we perform image captioning with an encoder-decoder structure, where the encoder is a
CNN and dense layer for images and words in a caption and the decoder is an RNN. We test several
different decoder variations, including different RNN cells, different input architectures, and different
hyperparameters. After this we caption images using a random generative and a deterministic method.
RNNs use state to represent time; in our model, we use an LSTM cell inside the RNN. The LSTM
uses various gates to memorize inputs, allowing it to memorize long term relationships in the sequence
data.

## 3: Experimentation and Comparison Between Pre-trained and Custom Convolution Neural Networks
A common adage in training neural networks says "Don’t be a hero", implying it is
easier to build neural networks from previous work than to train a model on ones
own. Here we experiment with both approaches to see which one achieves better
top-1 accuracy on our 20-category classification task. First we train a baseline
model and tune the model by modifying architecture and hyperparameters, starting
with 25% accuracy and improving to 45%. Next we train two state-of-the-art
models, VGG16 and Resnet18, freezing all but the last layer with accuracies of
74.06% and 77.89%.

## 2: Experimentation with Regularization, Activation, and Network architecture in Multi-layer Neural Networks to Improve Performance on Classification of SVHN Dataset
There are many methods to optimize Neural Network performance including
standardizing data, tuning hyperparameters, using momentum, early stopping,
regularization, different activation functions, and changing both dimensions of
network architecture. In this study we implement a Neural Network classifier
from scratch to classify the SVHN dataset with all of the mentioned methods, and
emphasis on experimental results of the last 3. We plot training and validation
losses and accuracies, then evaluate test accuracy, improving from 75% to upwards
of 84%.

## 1: Implementing Neural Networks from Scratch as a Means of Achieving High Accuracy
Many introductory neural networks leverage libraries without regard to the math behind the models. Here we show that it is possible to create models the perform well
from scratch given appropriate preprocessing. We apply principle componenent
analysis and create two classification models, Logistic and Softmax Regression
from scratch using NumPy. We then train both models on the Fashion MNIST
Dataset, achieving accuracies of 99%, 85% and 84% for all three tasks measured.
