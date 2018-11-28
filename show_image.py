import pandas as pd
import numpy as np
import os
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


data_dir = '../input'
img_dir = os.path.join(data_dir, 'bee_imgs')
data_csv = os.path.join(data_dir, 'bee_data.csv')
data = pd.read_csv(data_csv)

def to_file_path(file_name):
    return os.path.join(img_dir, file_name)

print(data.head())

img_wid = 120
img_len = 120
img_channels = 3 #RGB

#show image
def show_img(file):
    img = skimage.io.imread(os.path.join(img_dir, file))
    img = skimage.transform.resize(img, (img_wid, img_len), mode='reflect')

    return img[:,:,:img_channels]

#plt.imshow(img)
#plt.show()

test = data.iloc[0:2]
test1 = np.stack(test['file'].apply(show_img))
print(test1)

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

generator.fit(test1)