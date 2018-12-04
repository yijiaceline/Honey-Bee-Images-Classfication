import pandas as pd
import numpy as np
import os
import skimage
import random
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


img_dir = os.path.join('../input', 'bee_imgs')
data_csv = os.path.join('../input', 'bee_data.csv')
data = pd.read_csv(data_csv)

def to_file_path(file_name):
    return os.path.join(img_dir, file_name)

img_wid = 120
img_len = 120
img_channels = 3 #RGB

#show image
def show_img(file):
    img = skimage.io.imread(os.path.join(img_dir, file))
    img = skimage.transform.resize(img, (img_wid, img_len), mode = 'reflect')
    
    return img[:,:,:img_channels]

img = skimage.io.imread(os.path.join(img_dir, random.choice(data["file"])))
plt.imshow(img)
plt.show()
