
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
