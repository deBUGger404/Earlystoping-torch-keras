#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import confusion_matrix,accuracy_score


# In[2]:


#take mnist data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[3]:


x_train1 = x_train / 255
x_test1 = x_test / 255

x_train2 = np.expand_dims(x_train1.astype("float32"),-1)
x_test2 = np.expand_dims(x_test1.astype("float32") ,-1)


# In[4]:


y_train1 = tf.keras.utils.to_categorical(y_train, 10)
y_test1 = tf.keras.utils.to_categorical(y_test, 10)


# In[5]:


inp = tf.keras.Input(shape=(28,28,1))
x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu")(inp)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(10, activation="softmax")(x)


# In[6]:


model = tf.keras.Model(inputs=inp, outputs=x)


# In[7]:


model.summary()


# In[8]:


es = tf.keras.callbacks.EarlyStopping( monitor="val_loss", patience=2, verbose=1, restore_best_weights=True)
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.0005), metrics=["accuracy"])

out = model.fit(x_train2, y_train1, batch_size=128, epochs=50, validation_data=(x_test2,y_test1),verbose=1,callbacks=[es])


# In[9]:


train_loss = out.history['loss']
test_loss = out.history['val_loss']
plt.plot(train_loss)
plt.plot(test_loss)
plt.show()


# In[10]:


y_pred = model.predict(x_test2)
y_pred = (y_pred > 0.5) 


# In[11]:


print('Accuracy on test data: {:0.2f}'.format(accuracy_score(y_test1, y_pred)))


# In[ ]:




