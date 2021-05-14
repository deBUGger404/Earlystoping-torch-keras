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
- [Python-Script](#python-script)

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
  ### Pytorch:
  Below codes, use for early-stopping in PyTorch to overcome the model overfitting.
    ```python
    #use the ./pytorch/utils.py
    #as you can see the below code which will monitor the performance during the training
        elif score < self.best_score + self.delta:
          self.counter += 1
          self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
          if self.counter >= self.patience:
              self.early_stop = True
    # below code save model checkpoint while model will not imporving anymore
        def save_checkpoint(self, val_loss, model):
          if self.verbose:
              self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
          torch.save(model, self.path)
          self.val_loss_min = val_loss
    ```
   ### Tensorflow-Keras
    Below code, use for early-stoping for better performance in Keras.
    ```python
    import tensorflow as tf
    es = tf.keras.callbacks.EarlyStopping( monitor="val_loss", patience=2, verbose=1, restore_best_weights=True)
    ```

 ## Python-Script
 ```python
  # Start training with: 
  # pytorch
  python earlystoping_pytorch.py

  #tensorflow-keras
  python keras_early_stoping.py
 ```
 # Give a :star: to this Repository!
  
