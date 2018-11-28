import pandas as pd
import numpy as np
import os
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


data_dir = '../input'
img_dir = os.path.join(data_dir, 'bee_imgs')
data_csv = os.path.join(data_dir, 'bee_data.csv')
data = pd.read_csv(data_csv)

def to_file_path(file_name):
    return os.path.join(img_dir, file_name)

print(data.head())

img_wid = 100
img_len = 100
img_channels = 3 #RGB

#show image
def show_img(file):
    img = skimage.io.imread(os.path.join(img_dir, file))
    img = skimage.transform.resize(img, (img_wid, img_len), mode='reflect')

    return img[:,:,:img_channels]

plt.imshow(img)
plt.show()




