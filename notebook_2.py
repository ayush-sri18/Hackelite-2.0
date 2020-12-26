#!/usr/bin/env python
# coding: utf-8

# ## Pneumonia detection on chest X-rays

# Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.Symptoms typically include some combination of productive or dry cough, chest pain, fever and difficulty breathing. The severity of the condition is variable. Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases. Risk factors include cystic fibrosis, chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke and a weak immune system. Diagnosis is often based on symptoms and physical examination.Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis. The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.

# ### Importing the necessary libraries

# In[1]:


import tensorflow as tf
import keras
from keras import Input
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam,SGD,RMSprop
import os
from os import listdir, makedirs, getcwd, remove
import numpy as np
import pandas as pd
import glob2
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os
import scipy
import skimage
from skimage.transform import resize
import glob
import h5py
import shutil
import seaborn as sns
import cv2
import random as rn
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print(os.listdir('C:/Users/Abhishek Gupta/Desktop/X-ray dataset/17810_23812_bundle_archive (1)/chest_xray/chest_xray/train'))


# The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

# preparing dataset

# In[3]:


# here we have checked type of our images in our dataset.
#Since we are inputting 3 channels in our model so,images in our dataset must have 3 channels i.e.,RGB images.
img_name = 'IM-0117-0001.jpeg'
img_normal = load_img('C:/Users/Abhishek Gupta/Desktop/X-ray dataset/17810_23812_bundle_archive (1)/chest_xray/chest_xray/train/NORMAL/' + img_name)
img = cv2.imread('C:/Users/Abhishek Gupta/Desktop/X-ray dataset/17810_23812_bundle_archive (1)/chest_xray/chest_xray/train/NORMAL/' + img_name)
print(img.shape)
print('NORMAL')
plt.imshow(img_normal)
plt.show()


# In[4]:


img_name = 'person63_bacteria_306.jpeg'
img_pneumonia = load_img('C:/Users/Abhishek Gupta/Desktop/X-ray dataset/17810_23812_bundle_archive (1)/chest_xray/chest_xray/train/PNEUMONIA/' + img_name)
print('PNEUMONIA')
plt.imshow(img_pneumonia)
plt.show()


# In order to avoid overfitting problem, we need to expand artificially our dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations. Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more. By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.

# In[5]:


img_width, img_height = 120,120
train_dir = 'C:/Users/Abhishek Gupta/Desktop/X-ray dataset/17810_23812_bundle_archive (1)/chest_xray/train'
validation_dir ='C:/Users/Abhishek Gupta/Desktop/X-ray dataset/17810_23812_bundle_archive (1)/chest_xray/val'
test_dir = 'C:/Users/Abhishek Gupta/Desktop/X-ray dataset/17810_23812_bundle_archive (1)/chest_xray/test'


# In[6]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height,3)


# ### in continuation with notebook_2

# Data augmentation and normalisation to avoide overfitting

# In[7]:


batch_size=10
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[8]:


test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(imefg_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[9]:


test_generator = test_datagen.flow_from_directory(
    test_dir,
    class_mode='binary')


# Since the target dataset is small, it is not a good idea to fine-tune the ConvNet due to the risk of overfitting. Since the target data is similar to the base data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, we:
# 
# Remove the fully connected layers near the end of the pretrained base ConvNet
# Add a new fully connected layer that matches the number of classes in the target dataset
# Randomize the weights of the new fully connected layer and freeze all the weights from the pre-trained network
# Train the network to update the weights of the new fully connected layers

# In[11]:


from keras.applications.vgg16 import VGG16
base_model=VGG16(include_top=False, weights='imagenet', input_shape=(120,120,3), pooling='avg')


# In[12]:


model=Sequential()
model.add(base_model)
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))


for layer in base_model.layers[:15]:
    layer.trainable=False
for layer in base_model.layers[15:]:
    layer.trainable=True

model.summary()
model.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy'])


# ### in continuation with notebook_3

# In[ ]:





# In[ ]:





# In[ ]:




