
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms



# # EDA

# In[3]:


bee=pd.read_csv("bee_data.csv")


# In[4]:

'''
bee.shape


# In[5]:


bee.head()


# In[7]:


bee.dtypes


# In[27]:


bee.isnull().sum()


# In[25]:


bee["subspecies"].unique()


# In[36]:


#imbalance 
bee["subspecies"].value_counts()


# In[47]:


plt.figure(figsize=(16,5))
plt.bar(bee["subspecies"].value_counts().index, bee["subspecies"].value_counts(), )
plt.show()


# In[30]:


bee['health'].value_counts()


# In[173]:


plt.figure(figsize=(16,5))
plt.bar(bee['health'].value_counts().index, bee["health"].value_counts(), )
plt.show()
'''

# # Consider resampling?
# #### oversample undersample 
#     

# In[13]:


#!pip install nltk


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
x_train, x_test, y_train, y_test = train_test_split(bee["file"],bee["subspecies"],test_size=0.3,random_state=0)


# In[24]:


x_train.shape


# In[ ]:


import skimage
import skimage.io
import os
import skimage.transform


# In[200]:


img_dir="/home/ubuntu/Desktop/Final_Project/honey-bee-annotated-images/bee_imgs"
img = skimage.io.imread(os.path.join(img_dir, '001_047.png'))
plt.imshow(img)
plt.show()


# In[204]:


#RGB VALUES 
img


# In[202]:


img.shape



# In[23]:


img_wid = 120
img_len = 120
img_channels = 3 #RGB

#get image
def show_img(file):
    img = skimage.io.imread(os.path.join(img_dir, file))
    img = skimage.transform.resize(img, (img_wid, img_len), mode='reflect')

    return img[:,:,:img_channels]

#tensor
train_img = np.stack(x_train.apply(show_img))
print(train_img.shape)


# In[233]:

#tensor
test_img = np.stack(x_test.apply(show_img))


# In[26]:

from keras.preprocessing.image import ImageDataGenerator


# In[180]:

generator = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)

generator.fit(train_img)


# ## Two ways encoding label 

# In[171]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
#tensor
encoded_Y = le.transform(y_train)
encoded_yt = le.transform(y_test)
#arget = np_utils.to_categorical(encoded_Y )


# In[206]:


encoded_Y.shape


# In[161]:


nb_classes=7


# In[207]:

from keras.utils import np_utils
y_tr = np_utils.to_categorical(encoded_Y,nb_classes)
y_te=np_utils.to_categorical(encoded_yt,nb_classes)


# In[209]:


y_tr.shape


# In[150]:


#y_r = pd.get_dummies(y_train)



# In[226]:


from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization


# In[227]:


def cnn():
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    '''
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    '''
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    # dense layer with 50 neurons
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(7, activation = 'softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model = cnn()



# In[ ]:


def cnn_2():
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    # dense layer with 50 neurons
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(7, activation = 'softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model = cnn()


# In[228]:


model.summary()


# In[241]:


training1=model.fit(train_img, y_tr, batch_size = 30, validation_split = 0, epochs = 50, verbose = 1)


# In[235]:


training2= model.fit_generator(generator.flow(train_img,y_tr, batch_size=32)
                        ,epochs=20, verbose = 1
                        ,steps_per_epoch=50
                    )


# In[243]:


results = model.evaluate(test_img, y_te)
print('Test accuracy: ', results[1])


# In[244]:


pp=model.predict(test_img)


# In[246]:


y_pred = np.argmax(pp, axis=1)


# In[247]:


yyy=le.inverse_transform(encoded_yt)


# In[248]:


yyyy=le.inverse_transform(y_pred)


# In[249]:


from sklearn.metrics import classification_report
print(classification_report(yyy, yyyy))

