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
trainingSSa=modelS.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,sample_weight=Sweights,callbacks=[stop1])
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


# ## Adding normalization layer 
# #### Accuracy decreased at first, 69% 
# #### Adding one dropout layer did not improve much 74%
# #### adding more dropout layer increase 
# #### also tried adding L2 reguleration   

# In[114]:


def normal_cnn():
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2)
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(7, activation = 'softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model_normal = normal_cnn()


# In[115]:


start=time.time()#just batchN
training_normal=model_normal.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1)
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[162]:


results_normal = model_normal.evaluate(test_img, y_te)#just batchN
print('Test accuracy for deeper cnn with normalization layer: ', results_normal[1])


# In[37]:


def dropout_normal_cnn1():#adding dropout 
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(7, activation = 'softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model43_b = dropout_normal_cnn1()


# In[36]:


def dropout_normal_cnn2():#using he_normal 3 conv
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]),kernel_initializer='he_normal', filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(7, activation = 'softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model33_b = dropout_normal_cnn2()


# In[42]:


#model33_b.summary()


# In[167]:


from keras import regularizers
def dropout_ncnn3():# Adding regularizers
    model = Sequential()
    model.add(Conv2D(input_shape = (train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 50, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(7, activation = 'softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model43_bdr = dropout_ncnn3()


# In[156]:


start=time.time()# 43 adding dropout batchNorm
training43b=model43_b.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[157]:


results43b = model43_b.evaluate(test_img, y_te)#batch dropout 43
print('Test accuracy for deeper cnn 43 with normalization layer and dropout: ', results43b[1])


# In[158]:


show_report(encoded_yt,test_img,model43_b) #batch dropout 43


# In[159]:


get_kappa(encoded_yt,test_img,model43_b)#batch dropout 43


# In[168]:


start=time.time()#weight regularizer
training43bdr=model43_bdr.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[169]:


results43bdr = model43_bdr.evaluate(test_img, y_te)#weight regularizer
print('Test accuracy for deeper cnn 43 with normalization layer,drop and kernel regularizer: ', results43bdr[1])


# In[170]:


show_report(encoded_yt,test_img,model43_bdr)#weight regularizer


# In[171]:


get_kappa(encoded_yt,test_img,model43_bdr)#weight regularizer


# In[161]:


start=time.time()#3conv he_normal batch dropout
training33b=model33_b.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[162]:


results33b = model33_b.evaluate(test_img, y_te)
print('Test accuracy for deeper cnn 33 with normalization layer and dropout: ', results33b[1])


# In[163]:


show_report(encoded_yt,test_img,model33_b)##3conv he_normal batch dropout


# In[164]:


get_kappa(encoded_yt,test_img,model33_b)#3conv he_normal batch dropout


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





# ## Changing Learning rate

# In[50]:


def cnn_43_lr005():#lr 0.005
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
    adam = optimizers.Adam(lr = 0.005)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model43lr005 = cnn_43_lr005()


# In[51]:


start=time.time()#lr 0.005
training43lr005=model43lr005.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1)
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[53]:


results43lr005 = model43lr005.evaluate(test_img, y_te)
print('Test accuracy for deeper cnn 43 with lr 0.005: ', results43lr005[1])


# In[54]:


show_report(encoded_yt,test_img,model43lr005)#lr 0.005


# In[55]:


get_kappa(encoded_yt,test_img,model43lr005)#lr 0.005


# In[56]:


def cnn_43_lr0007():#lr 0.0007
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
    adam = optimizers.Adam(lr = 0.0007)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model    

model43lr0007 = cnn_43_lr0007()


# In[57]:


start=time.time()#lr 0.0007
training43lr0007=model43lr0007.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1)
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[58]:


results43lr0007 = model43lr0007.evaluate(test_img, y_te)
print('Test accuracy for deeper cnn 43 with lr 0.0007: ', results43lr0007[1])


# In[59]:


show_report(encoded_yt,test_img,model43lr0007)#lr 0.0007


# In[60]:


get_kappa(encoded_yt,test_img,model43lr0007)#lr 0.0007


# In[67]:


def scheduler(epoch):
    if epoch%5==0 and epoch!=0:
        lr = K.get_value(model_SS.optimizer.lr)
        K.set_value(model_SS.optimizer.lr, lr*.75)
        print("lr changed to {}".format(lr*.75))
    return K.get_value(model_SS.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)


# In[68]:
def basic_cnn_SS2():#using sgd
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

model_SS2 = basic_cnn_SS2()


start=time.time()
trainingSSlrchaning=model_SS2.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1,lr_decay])
end=time.time()
print("--------------------------")
print("Total training time (seconds)", end-start)


# In[70]:


results43lrchanging = model_SS2.evaluate(test_img, y_te)
print('Test accuracy for deeper cnn 43 with lr changing for sgd: ', results43lrchanging[1])


# In[76]:


show_report(encoded_yt,test_img,model_SS2)#lrchanging sgd


# In[77]:


get_kappa(encoded_yt,test_img,model_SS2)#lrchanging sgd


# In[ ]:


## Final_model with tsb


# In[80]:
def cnn_43gf(): #model43
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

model43f = cnn_43gf()

training_finalC=model43f.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1,tsb])


# In[85]:


plt.plot(training_finalC.history['loss'])
plt.title("Model43 loss")


# In[84]:


show_report(encoded_yt,test_img,model43)


# In[96]:


y43_pred=model43.predict(test_img)


# In[82]:


plt.plot(training_finalC.history['acc'],label="training acc")
plt.plot(training_finalC.history['val_acc'],label="validation acc")
plt.legend(loc='best')
plt.title("model 43 ACC ")


# In[111]:
def cnn_43ff():
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

model43_ff = cnn_43ff()

training_finalF_f=model43_ff.fit(train_img, y_tr, batch_size = 32, validation_split = 0.2, epochs = 50, verbose = 1,callbacks=[stop1])


# In[112]:


plt.plot(training_finalF_f.history['acc'],label="training acc")
plt.plot(training_finalF_f.history['val_acc'],label="validation acc")
plt.legend(loc='best')
plt.title("model focal loss ACC ")


# In[113]:


plt.plot(training_finalF_f.history['loss'])
plt.title("Model focal loss")


# In[107]:


def show_roc_auc(model): 
    y_pred=model.predict(test_img)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_te[:,i], y_pred[:,i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    colors = (['blue', 'red', 'green',"yellow","pink","blue","orange","gray","violet","brown"])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(clas[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--',)
    plt.xlim([0, 1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for all labels')
    plt.legend(loc=9,bbox_to_anchor=(0.5,-0.2))
    plt.show()


# In[108]:


show_roc_auc(model43)


# In[114]:


show_roc_auc(model43_f)


# In[94]:


def visualize_kernel(layer_idx,img):
    layer = model43.layers[layer_idx]
    layer_weight=layer.get_weights()[0]
    nb_kernel=layer_weight.shape[3]
    title="conv{} layer kernal".format(layer_idx)
    fig, axes =  plt.subplots(nb_kernel, 3, figsize=(8, nb_kernel*2))
    for i in range(0, nb_kernel):
        # Get kernel from the layer and draw it
        kernel=layer_weight[:,:,:3,i]
        axes[i][0].imshow((kernel* 255).squeeze().astype(np.uint8), vmin=0, vmax=255)
        axes[i][0].set_title("Kernel %d" % i, fontsize = 7)
        
        # Get and draw sample image from test data
        axes[i][1].imshow((img * 255).astype(np.uint8), vmin=0, vmax=255)
        axes[i][1].set_title("Before", fontsize=8)
        
        # adding weight 
        img_filt = scipy.ndimage.filters.convolve(img, kernel)
        axes[i][2].imshow((img_filt * 255).astype(np.uint8), vmin=0, vmax=255)
        axes[i][2].set_title("After", fontsize=8)
        
    plt.suptitle(title)
    plt.tight_layout( )
    plt.subplots_adjust(top=0.95)
    plt.show() 


# In[95]:


visualize_kernel(0,test_img[2,:,:,:])


# In[ ]:




