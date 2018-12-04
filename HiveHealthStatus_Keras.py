import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bees = pd.read_csv('../input/bee_data.csv')
'''  
plt.figure(figsize=(6,6))
bees.health.value_counts().plot(kind = 'bar')
plt.title('Hive Health')
plt.show()
'''
unhealthy = bees.loc[bees['health'] != 'healthy']
#unhealthy
unhealthy.count()

healthy = bees.loc[bees['health'] == 'healthy']
healthy.count()

bees.health.value_counts()

New = bees.replace(['hive being robbed', 'few varrao, hive beetles', 'Varroa, Small Hive Beetles', 'ant problems', 'missing queen'],
             'unhealthy')

New.head()

from sklearn.model_selection import train_test_split
import skimage
import skimage.io
import os
import skimage.transform

np.random.seed(1)

x_train, x_test, y_train, y_test = train_test_split(New["file"],New["health"],test_size=0.3,random_state=0)

#x_train.shape

img_dir="../input/bee_imgs"

img_wid = 120
img_len = 120
img_channels = 3 #RGB

# get image
def show_img(file):
    img = skimage.io.imread(os.path.join(img_dir, file))
    img = skimage.transform.resize(img, (img_wid, img_len), mode='reflect')

    return img[:, :, :img_channels]

train_img = np.stack(x_train.apply(show_img))
print(train_img.shape)

test_img = np.stack(x_test.apply(show_img))

from keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=180,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

generator.fit(train_img)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_train)

encoded_y = le.transform(y_train)
encoded_ytest = le.transform(y_test)

#encoded_Y.shape

Bee_classes = 7

# Define Class, Convert 1-dimensional class arrays to 7-dimensional class matrices
from keras.utils import np_utils

y_train = np_utils.to_categorical(encoded_y, Bee_classes)
y_test = np_utils.to_categorical(encoded_ytest, Bee_classes)


from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization

#model architecture
def cnn():
    model = Sequential()
    model.add(
        Conv2D(input_shape=(train_img.shape[1], train_img.shape[2], train_img.shape[3]), filters=50, kernel_size=(3, 3),
               strides=(1, 1), padding='same', kernel_initializer='he_normal'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # dense layer with 50 neurons
    model.add(Dense(50, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    adam = optimizers.Adam(lr=0.001)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

model = cnn()

model.summary()

#fit trainning data model
training1 = model.fit(train_img, y_train, batch_size=64, validation_split=0.2, epochs=30, verbose=1)

training2 = model.fit_generator(generator.flow(train_img, y_train, batch_size=32)
                                , epochs=30, verbose=1)
#print(training2.history.keys())

#accuracy = training1.history['acc']
#loss = training1.history['loss']

accuracy = training2.history['acc']
loss = training2.history['loss']

epochs = range(1, len(accuracy) + 1)

plt.plot(accuracy)
plt.title('Hive Health Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(loss)
plt.title('Hive Health Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

#evaluate test data
results = model.evaluate(test_img, y_test)
print('Test accuracy: ', results[1])

pred_test = model.predict(test_img)
y_pred = np.argmax(pred_test, axis=1)

label_ytest = le.inverse_transform(encoded_ytest)
label_y_pred = le.inverse_transform(y_pred)

from sklearn.metrics import classification_report
print(classification_report(label_ytest, label_y_pred))