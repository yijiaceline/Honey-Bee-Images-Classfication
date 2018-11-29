import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


bee=pd.read_csv("bee_data.csv")

x_train, x_test, y_train, y_test = train_test_split(bee["file"],bee["subspecies"],test_size=0.3,random_state=0)

img_dir="/home/ubuntu/Desktop/Final_Project/honey-bee-annotated-images/bee_imgs"
img = skimage.io.imread(os.path.join(img_dir, '001_047.png'))
plt.imshow(img)
plt.show()