#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import random
from sklearn.metrics import classification_report
from keras import callbacks
import random 
import scipy
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from keras import backend as K
from sklearn.metrics import roc_auc_score


# In[2]:


np.random.seed(100)


# In[4]:


bee=pd.read_csv("../input/bee_data.csv")


# # EDA

# In[4]:


bee.shape


# In[5]:


bee.head()


# In[6]:


bee.dtypes


# In[7]:


bee.isnull().sum()


# In[8]:


bee["subspecies"].unique()


# ## Imbalanced data

# In[16]:


#imbalance 
bee["subspecies"].value_counts()


# In[14]:


plt.figure(figsize=(16,5))
plt.bar(bee["subspecies"].value_counts().index, bee["subspecies"].value_counts() )
plt.show()


# In[15]:


bee['health'].value_counts()


# # Consider resampling? Change accuracy metrix? 
# 

# ## Split dataset

# In[7]:


import skimage
import skimage.io
import os
import matplotlib.pyplot as plt
import skimage.transform


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bee["file"],bee["subspecies"],test_size=0.3,random_state=0)


# In[9]:


print(x_train.shape)
print(x_test.shape)


# In[10]:


#random showing an imag
img_dir="../input/bee_imgs"
img = skimage.io.imread(os.path.join(img_dir, random.choice(bee["file"])))
plt.imshow(img)
plt.show()


# In[11]:


#Check RGB VALUES 
#img


# In[12]:


#img.shape


# # Get image for trainset and testset 

# In[13]:


img_wid = 120
img_len = 120
img_channels = 3 

#get image
def show_img(file):
    img = skimage.io.imread(os.path.join(img_dir, file))
    img = skimage.transform.resize(img, (img_wid, img_len), mode='reflect')

    return img[:,:,:img_channels]


train_img = np.stack(x_train.apply(show_img))
print(train_img.shape)


# In[14]:


test_img = np.stack(x_test.apply(show_img))


# In[18]:


print(test_img.shape)


# ## Encoding label 

# In[16]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
encoded_Y = le.transform(y_train)
encoded_yt = le.transform(y_test)


# In[17]:


#y_rr = pd.get_dummies(y_train)


# In[19]:


encoded_Y.shape


# In[20]:


nb_classes=7


# In[21]:


from keras.utils import np_utils
y_tr = np_utils.to_categorical(encoded_Y,nb_classes)
y_te=np_utils.to_categorical(encoded_yt,nb_classes)


# In[22]:


y_tr.shape


# In[23]:


y_te.shape


# In[24]:


y_tr[0]


# In[25]:


from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization, Dropout


# In[26]:


from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler,TensorBoard
stop1=EarlyStopping(monitor='loss',  patience=5, verbose=0, mode='auto')




# In[28]:


tsb = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1,write_images=True, write_grads=True)


# In[32]:


def show_report(encoded_yt, test_img,model):
    y_pred=model.predict(test_img)
    y_pred=np.argmax(y_pred,axis=1)
    y_trues=le.inverse_transform(encoded_yt)
    y_tests=le.inverse_transform(y_pred)
    print( classification_report(y_trues, y_tests))


# In[33]:


def get_kappa(encoded_yt, test_img,model):
    y_pred=model.predict(test_img)
    y_pred=np.argmax(y_pred,axis=1)
    y_trues=le.inverse_transform(encoded_yt)
    y_tests=le.inverse_transform(y_pred)
    print( cohen_kappa_score(y_trues, y_tests))


# In[44]:


classes_name=bee["subspecies"].value_counts().index
classes_namet=le.transform(bee["subspecies"].value_counts().index)


# In[45]:

#create class dict for later use
clas=dict(zip(classes_namet,classes_name))


# # Building model

# ## Basic CNN
# #### Padding 0 &nbsp; Stride 1 &nbsp;  Loss fuc: categorical_crossentropy &nbsp; optimizer:adam &nbsp;  &nbsp; 2 Conv layer  &nbsp; kernel__size=[5,5,3,3]     &nbsp; ACC: 0.90  &nbsp;  kernel__size=[3,3,3,3] &nbsp; ACC: 0.92     

# In[29]:


def basic_cnn_S():#using softmax
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    #7 hidden neurons(fully connected)
    model.add(Dense(7, activation = 'softmax'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

modelS = basic_cnn_S()


# In[30]:


def basic_cnn_Sg():#using sigmoid
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    #7 hidden neurons(fully connected)
    model.add(Dense(7, activation = 'sigmoid'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

modelSg = basic_cnn_Sg()


# In[31]:


def basic_cnn_R():#using relu
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    #7 hidden neurons(fully connected)
    model.add(Dense(7, activation = 'relu'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

modelR = basic_cnn_R()


# In[28]:


#modelS.summary()


# In[29]:


#train softmax
start=time.time()
trainingS=modelS.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[29]:


start=time.time()#train sigmoid
trainingSg=modelSg.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[97]:


start=time.time()#20 epochs softmax
trainingS20=modelS.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 20, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[102]:


start=time.time() #30 epochs softmax
trainingS30=modelS.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 30, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[93]:


start=time.time()#relu
trainingR=modelR.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[103]:


resultsS = modelS.evaluate(test_img, y_te)
print('Test accuracy for basic cnn with softmax: ', resultsS[1])


# In[30]:


resultsSg = modelSg.evaluate(test_img, y_te)
print('Test accuracy for basic cnn with sigmoid: ', resultsSg[1])


# In[34]:


show_report(encoded_yt, test_img, modelSg)#sigmoid


# In[35]:


get_kappa(encoded_yt, test_img, modelSg)#sigmoid


# In[35]:


resultsR = modelR.evaluate(test_img, y_te)
print('Test accuracy for basic cnn with relu: ', resultsR[1])


# In[105]:


show_report(encoded_yt, test_img, modelS)#softmax


# In[37]:


show_report(encoded_yt, test_img, modelR)#relu


# In[106]:


get_kappa(encoded_yt, test_img,modelS)#softmax


# In[ ]:


get_kappa(encoded_yt, test_img,modelR)#relu



# In[63]:


def basic_cnn_SS():#using sgd
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    #7 hidden neurons(fully connected)
    model.add(Dense(7, activation = 'softmax'))
    sgd = optimizers.sgd(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    
    return model    

model_SS = basic_cnn_SS()


# In[66]:


start=time.time()#SGD OPTIMIZER
training_SS=model_SS.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[67]:


results_SS = model_SS.evaluate(test_img, y_te)
print('Test accuracy for basic cnn with sgd: ', results_SS[1])


# In[68]:


show_report(encoded_yt, test_img,model_SS)#sgd


# In[ ]:


get_kappa(encoded_yt, test_img,model_SS)#SGD


# In[56]:
#Try adding sample weights to train 

from sklearn.utils.class_weight import compute_sample_weight


# In[61]:


Sweights = compute_sample_weight('balanced',  y_tr)


# In[63]:

#sample weights
def basic_cnn_Sa():#using softmax
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    #7 hidden neurons(fully connected)
    model.add(Dense(7, activation = 'softmax'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

modelSa = basic_cnn_Sa()
start=time.time()
trainingSSa=modelSa.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,sample_weight=Sweights,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[69]:


resultsssa = modelSa.evaluate(test_img, y_te)
print('Test accuracy for basic cnn adding sample weights: ', resultsssa[1])


# In[70]:


show_report(encoded_yt, test_img, modelSa)#sample weights


# In[76]:


get_kappa(encoded_yt, test_img, modelSa)#sample weights