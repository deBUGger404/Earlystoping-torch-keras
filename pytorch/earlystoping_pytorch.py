#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import  models


# In[2]:


# !pip3 install torch==1.7


# In[3]:


warnings.filterwarnings("ignore")


# In[4]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=8)


# In[5]:


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[6]:


import torch.nn as nn
import torch.nn.functional as F


class model_es(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 20, 5)
        self.fc1 = nn.Linear(20 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[8]:


import torch.optim as optim
from utils import binary_acc, EarlyStopping, AverageMeter

model = model_es()
criterion = nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=15, verbose=True,path = 'model1.pth')
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[11]:


epochs = 50

train_acc = AverageMeter()
train_losses = AverageMeter()
val_acc = AverageMeter()
val_losses = AverageMeter()
valid_loss_min = np.Inf
for epoch in range(1,epochs):
    model.train()
    for inputs, labels in trainloader:
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs.float())
        loss = criterion(logps, labels)
        acc = binary_acc(logps, labels)
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), inputs.size(0))
        train_acc.update(acc, inputs.size(0))

    # VALIDATION
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs.float())
            val_loss  = criterion(logps, labels)
            val_ac  = binary_acc(logps, labels)
            val_losses.update(val_loss.item(), inputs.size(0))
            val_acc.update(val_ac, inputs.size(0))
    print(f'Epoch {epoch}: | Train Loss: {train_losses.avg:.5f} |     Val Loss: {val_losses.avg:.5f} | Train Acc: {train_acc.avg:.3f} |      Val Acc: {val_acc.avg:.3f}')
    early_stopping(val_losses.avg, model)


# In[13]:


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test: %d %%' % (100 * correct / total))


# In[ ]:




