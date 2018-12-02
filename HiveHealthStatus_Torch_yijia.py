import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import skimage
import skimage.io
from pandas import Series
import os
import skimage.transform
from PIL import Image
from time import time


data_dir = '../input'
img_dir = os.path.join(data_dir, 'bee_imgs')
data_csv = os.path.join(data_dir, 'bee_data.csv')
data = pd.read_csv(data_csv)

#unhealthy
unhealthy = data.loc[data['health'] != 'healthy']

healthy = data.loc[data['health'] == 'healthy']

data = data.replace(['hive being robbed', 'few varrao, hive beetles', 'Varroa, Small Hive Beetles', 'ant problems', 'missing queen'],
             'unhealthy')

def to_file_path(file_name):
    return os.path.join(img_dir, file_name)


#set subspecies
target = data['health']
target = Series.as_matrix(target)
target_list = set(target)
target_list = list(target_list)

dic = {}
for i in range(2):
    dic[target_list[i]] = i
data = data.replace({"health": dic})

#define dataset

class honeybee(Dataset):
    def __init__(self, data, transform = None):
        self.data = data
        self.img_dir = '../input/bee_imgs'
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img = os.path.join(self.img_dir, data.iloc[index,0])
        image = Image.open(img)
        image = image.resize((120,120))
        image = image.convert('RGB')
        image = self.transform(image)
        label = self.data.iloc[index, 6]
        label = torch.tensor(np.asarray(label))

        return image,label

    def __len__(self):
        return len(self.data)


# split train test
train_data, test_data = train_test_split(data, test_size=0.3)
train_data = honeybee(train_data)
test_data = honeybee(test_data)


epochs = 15
batch_size = 32
learning_rate = 0.01

#Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
'''
dataiter = iter(train_loader)
images, labels = dataiter.next()
'''

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride= 1, padding=2),  # RGB image channel = 3, output channel = num_filter
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride= 1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) #output size (64,61,61)

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) #output size(128,32,32)

        self.layer4 = nn.Sequential(
            nn.Linear(128*32*32, 2048),
            nn.Linear(2048, 128),
            nn.Linear(128, 2))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)

        return out

cnn = CNN()
cnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
start = time()
# Train the Model
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d],  Loss: %.4f'
                  % (epoch + 1, epochs, loss.item()))


# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

end = time()
print('Computational Time:', end - start)
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model on the 1000 test images: %d %%' % (100 * correct / total))