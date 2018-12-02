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
from torch.autograd import Variable
from pandas import Series
import os
from PIL import Image
from time import time


data_dir = 'input'
img_dir = os.path.join(data_dir, 'bee_imgs')
data_csv = os.path.join(data_dir, 'bee_data.csv')
data = pd.read_csv(data_csv)

def to_file_path(file_name):
    return os.path.join(img_dir, file_name)

#set subspecies
target = data['subspecies']
target = Series.as_matrix(target)
target_list = set(target)
target_list = list(target_list)

dic = {}
for i in range(7):
    dic[target_list[i]] = i
data = data.replace({"subspecies": dic})

#define dataset

class honeybee(Dataset):
    def __init__(self, data, transform = None):
        self.data = data
        self.img_dir = 'input/bee_imgs'
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img = os.path.join(self.img_dir, data.iloc[index,0])
        image = Image.open(img)
        image = image.resize((120,120))
        image = image.convert('RGB')
        image = self.transform(image)
        label = self.data.iloc[index, 5]
        label = torch.tensor(np.asarray(label))

        return image,label

    def __len__(self):
        return len(self.data)


# split train test
train_data, test_data = train_test_split(data, test_size=0.3)
train_data = honeybee(train_data)
test_data = honeybee(test_data)


epochs = 20
batch_size = 32
learning_rate = 0.001

#Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=2),  # RGB image channel = 3, output channel = num_filter
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride= 1, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Softmax2d(),
            nn.Dropout(p=0.5))

        '''
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2))
        '''
        self.fc1 = nn.Linear(61*61*64,64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


cnn = CNN()
cnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
start = time()
Loss = []
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
        Loss.append(loss.item())


plt.title("CrossEntropyLoss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(Loss)
plt.show()



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
print('Test Accuracy of the model on the 1552 test images: %d %%' % (100 * correct / total))