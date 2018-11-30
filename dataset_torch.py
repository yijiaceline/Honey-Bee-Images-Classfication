import pandas as pd
import numpy as np
import os
import skimage
import skimage.io
import skimage.transform
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pandas import Series
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

data_dir = '../input'
img_dir = os.path.join(data_dir, 'bee_imgs')
data_csv = os.path.join(data_dir, 'bee_data.csv')
data = pd.read_csv(data_csv)

def to_file_path(file_name):
    return os.path.join(img_dir, file_name)

img_wid = 120
img_len = 120
img_channels = 3 #RGB

#get image
def show_img(file):
    img = skimage.io.imread(os.path.join(img_dir, file))
    img = skimage.transform.resize(img, (img_wid, img_len), mode='reflect')

    return img[:,:,:img_channels]


#set subspecies
target = data['subspecies']
target = Series.as_matrix(target)
target_list = set(target)
target_list = list(target_list)

dic = {}
for i in range(7):
    dic[target_list[i]] = i



# img_img = data["file"]
# img_path = [os.path.join(img_dir, i) for i in img_img]


#
# #get file path
# x_train_img = [os.path.join(img_dir, i) for i in x_train]
# x_test_img = [os.path.join(img_dir, i) for i in x_test]
# #make it to list
# y_train_list = Series.as_matrix(y_train)
# y_test_list = Series.as_matrix(y_test)


#define dataset
transform = transforms.Compose(transforms.ToTensor())

class honeybee(Dataset):
    def __init__(self, data, img_dir, transform = None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img = os.path.join(self.img_dir, data.iloc[index,0])
        image = Image.open(img)
        image = image.convert('RGB')
        label = self.self.data.iloc[index, 5]
        return image,label

    def __len__(self):
        return len(self.images)


# split train test
train_data, test_data = train_test_split(data, test_size=0.3)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4)
# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
#
# imshow(utils.make_grid(images))
# plt.show()
# #----------
# # Check for cuda
# device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
# print(device)