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


# ## Adding layers 

# #### kernel__size=[5,5,3,3,3,3,2,2] &nbsp; 4 CONV

# In[35]:


def cnn_45():
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (5,5), strides = (1,1), padding = 'same',name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv2"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv3"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (2,2), strides = (1,1), padding = 'same',name="conv4"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu',name='dense'))
    model.add(Dense(7, activation = 'softmax',name='preds'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model45 = cnn_45()


# In[74]:


#model45.summary()


# In[79]:


start=time.time()#model45
training45=model45.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[81]:


results45 = model45.evaluate(test_img, y_te)
print('Test accuracy for deeper cnn 45: ', results45[1])


# In[85]:


show_report(encoded_yt, test_img, model45)#model45


# In[87]:


get_kappa(encoded_yt, test_img, model45)


# ## Try other kernel size

# #### kernel__size=[3,3,3,3,3,3,2,2] &nbsp; 4 CONV

# In[78]:


def cnn_43g(): #model43
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv2"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv3"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv4"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu',name='dense'))
    model.add(Dense(7, activation = 'softmax',name='preds'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model43 = cnn_43g()


# In[89]:


start=time.time()#3,3,3,3,2,2 model43
training43=model43.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[90]:


results43 = model43.evaluate(test_img, y_te)#model43
print('Test accuracy for deeper cnn 43: ', results43[1])


# In[91]:


show_report(encoded_yt, test_img, model43)#3,3,3,3,2,2 model43


# In[92]:


get_kappa(encoded_yt, test_img, model43)#model43


# In[72]:

#tey another batch size
def cnn_4364(): #model43
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv2"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv3"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv4"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu',name='dense'))
    model.add(Dense(7, activation = 'softmax',name='preds'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model4364 = cnn_4364()
start=time.time()#3,3,3,3,2,2
training4364=model4364.fit(train_img, y_tr, batch_size = 64, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[73]:


results4364 = model4364.evaluate(test_img, y_te)#batch size64
print('Test accuracy for deeper cnn 43 with 64 batch_size: ', results4364[1])


# In[74]:


show_report(encoded_yt, test_img, model4364)#64


# In[75]:


get_kappa(encoded_yt, test_img, model4364)#64


# # Changing weight intialization

# In[107]:


def cnn_43h(): #using he_normal
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv2"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv3"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv4"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    #dense layer with 50 neurons
    model.add(Dense(50, activation = 'relu',name='dense'))
    model.add(Dense(7, activation = 'softmax',name='preds'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model43_h = cnn_43h()


# In[108]:


start=time.time()#he_normal
training43h=model43_h.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[110]:


results43h = model43_h.evaluate(test_img, y_te)#he_normal
print('Test accuracy for deeper cnn 43_h: ', results43h[1])


# In[114]:


show_report(encoded_yt, test_img, model43_h)#he_normal


# In[115]:


get_kappa(encoded_yt, test_img, model43_h)#he_normal


# # Adding dropout layer

# In[194]:


def cnn_43_drop():#one dropout layer
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv2"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv3"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv4"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    #dense layer with 50 neurons
    model.add(Dense(50, activation = 'relu',name='dense'))
    model.add(Dense(7, activation = 'softmax',name='preds'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model43_d= cnn_43_drop()


# In[ ]:


start=time.time()#dropout
training43d=model43_d.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[123]:


results43d = model43_d.evaluate(test_img, y_te)
print('Test accuracy for deeper cnn 43 with dropout: ', results43d[1])


# In[124]:


show_report(encoded_yt,test_img,model43_d)#drop out


# In[125]:


get_kappa(encoded_yt,test_img,model43_d)#dropout



# ## Change loss function

# ## weighted_categorical_crossentropy
# ## Refenrence: (copyed all ) &nbsp; https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d

# In[173]:


from keras import backend as K
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


# In[174]:


def cnn_43w():#weighted_categorical_crossentropy
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same', name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv2"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv3"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv4"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu',name='dense'))
    model.add(Dense(7, activation = 'softmax',name='preds'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = weighted_categorical_crossentropy(weights), optimizer = adam, metrics = ['accuracy'])
    
    return model    

model43_w = cnn_43w()


# In[175]:

#weighted_categorical_crossentropy
start=time.time()
training43w=model43_w.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[176]:


results43w = model43_w.evaluate(test_img, y_te)
print('Test accuracy for deeper cnn 43 using weighted loss CE: ', results43w[1])


# In[178]:


show_report(encoded_yt,test_img,model43_w)#weighted_categorical_crossentropy


# In[179]:


get_kappa(encoded_yt,test_img,model43_w)#weighted_categorical_crossentropy


# ##  focal loss

# ## Refenrence: (copyed all )   https://github.com/mkocabas/focal-loss-keras

# In[48]:


from keras import backend as K
import tensorflow as tf
'''
Compatible with tensorflow backend
'''
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed


# In[110]:


def cnn_43f():
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv2"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv3"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',name="conv4"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    #dense layer with 50 neurons
    model.add(Dense(50, activation = 'relu',name='dense'))
    model.add(Dense(7, activation = 'sigmoid',name='preds'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = [focal_loss(alpha=.25, gamma=2)], optimizer = adam, metrics = ['accuracy'])
    
    return model    

model43_f = cnn_43f()


# In[190]:


start=time.time()#focal loss
training43f=model43_f.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1)
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[191]:


results43f = model43_f.evaluate(test_img, y_te)
print('Test accuracy for deeper cnn 43 with focal loss: ', results43f[1])


# In[192]:


show_report(encoded_yt,test_img,model43_f)#focal loss


# In[193]:


get_kappa(encoded_yt,test_img,model43_f)#focal loss