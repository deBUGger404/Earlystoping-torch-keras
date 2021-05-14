# earlystoping-torch-keras

<p align="center">
<img src="https://user-images.githubusercontent.com/59862546/117540979-59b64100-b02f-11eb-9ea9-457ecf2e2271.png" width="400" height="200"> 
<img src="https://blog.keras.io/img/keras-tensorflow-logo.jpg" width="400" height="200"> 
<p>
  
  
## Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Training](#training)
  - [Pytorch](#pytorch)
  - [Tensorflow-Keras](#tensorflow-keras)

## Introduction
- In PyTorch training, I used the `CIFAR-10` dataset which consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
- In TensorFlow-Keras training, I used MNIST data which consists of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

## Prerequisites
- Python>=3.6
- PyTorch >=1.4
- Tensorflow>=2.0
- Library are mentioned in `requirenments.txt`

## Training
- If We train our model with too many epochs then it will start overfitting on the training dataset and showing worse performance on the test dataset. And vice versa, too few epochs can lead the model to underfit on trainset.
- So, Early-stopping is an approach to stop training after some epoch if there is n significant improvement in performance.
- Basically, Early-stopping monitors the performance during the training using TensorFlow-Keras API.
- If you use early-stoping in Pytorch, below is the approch:
  ```python
  use the ./pytorch/utils.py
  as you can see the below code which will monitor the performance during the training
          elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
  ```
