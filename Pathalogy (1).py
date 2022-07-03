#!/usr/bin/env python
# coding: utf-8

# ### My-Code 
# 

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
from torchvision import datasets, models
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import datasets, transforms
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
torch.use_deterministic_algorithms(True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## UNZIP the pathalogy.zip file extracted from LMS 
# ### After unzipping we will get the directory name pathalogyData , inside that we have another directory cancer data . Once this done then provide the path of "pathalogyData " in the below code .
# ### Please note with this code we are creating sub directories for test and train 

# In[3]:


import os
import numpy as np
import shutil
rootdir = r"C:\Users\RajputD2\Working_Repository\Dushyant_personal\ISB-Directory\Term-4\Deep_Learning\pathologyData\cancerData"
 #path of the original folder

classes = ['background', 'foreground']

for i in classes:
    os.makedirs(rootdir +'/train/' + i)
    os.makedirs(rootdir +'/test/' + i)
    source = rootdir + '/' + i
    allFileNames = os.listdir(source)
    np.random.shuffle(allFileNames)
    test_ratio = 0.20
    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                               [int(len(allFileNames)* (1 - test_ratio))])
    
    train_FileNames = [source+'/'+ name for name in train_FileNames.tolist()]
    test_FileNames = [source+'/' + name for name in test_FileNames.tolist()]

    for name in train_FileNames:
        shutil.copy(name, rootdir +'/train/' + i)

    for name in test_FileNames:
        shutil.copy(name, rootdir +'/test/' + i)
    


# ## NOTE: FROM THE ABOVE CODE TWO DIRECTORIES WILL BE CREATED 
# #### same needs to be mentioend below 

# In[4]:


data_dir = r"C:\Users\RajputD2\Working_Repository\Dushyant_personal\ISB-Directory\Term-4\Deep_Learning\pathologyData\cancerData\train"
test_data_dir = r"C:\Users\RajputD2\Working_Repository\Dushyant_personal\ISB-Directory\Term-4\Deep_Learning\pathologyData\cancerData\test"


# In[ ]:


#dataset1 = ImageFolder(data_dir,transform = transforms.Compose([transforms.ToTensor()]))


# In[5]:


#load the train and test data
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))


# In[ ]:


#dataset = ImageFolder(rootdir,transform = transforms.Compose([
#    transforms.Resize((150,150)),transforms.ToTensor()
#]))
#dataset


# In[6]:


dataset


# In[7]:


test_dataset


# In[8]:


print("Follwing classes are there : \n",dataset.classes)


# In[18]:


def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))

#display any image in the dataset
display_img(*dataset[1805])


# In[25]:


batch_size = 128
dataloader_train = DataLoader(dataset, batch_size=batch_size,shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size,shuffle=True )


# In[26]:


dataloader_test


# In[ ]:


#from torch.utils.data.dataloader import DataLoader
#from torch.utils.data import random_split

#batch_size = 128
#val_size = 250
#train_size = len(dataset) - val_size 

#train_data,val_data = random_split(dataset,[train_size,val_size])
#print(f"Length of Train Data : {len(train_data)}")
#print(f"Length of Validation Data : {len(val_data)}")


#load the train and validation into batches.
#dataloader_train = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
#dataloader_test = DataLoader(val_data, batch_size, num_workers = 4, pin_memory = True)




# In[24]:


for image,label in dataloader_train:
    print("Image shape: ",image.shape)
    print("Image tensor: ", image)
    print("Label: ", label)
    break


# In[27]:


from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break
        
show_batch(dataloader_train)


# In[28]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 3 input image channel (RGB), 2 output channels/feature maps
        # 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 16, 3,padding=1) # 16 filters of 3x3 size,depth 1,pad 1 and stride 1
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,8,3, padding=1)# 8 filters of 3x3 size,depth 1,pad 1 and stride 1
        #x = x.flatten(start_dim=1)
        self.fc1 = nn.Linear(37*37*8,64) # Dense layer of size 64
        #self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64,2)
        
    def forward(self, x):
        
         # BLOCK 1: CONV + RELU +MAXPOOL (POOL 2x2 with stride 2)
        x = self.pool(F.relu(self.conv1(x)))
        
        # BLOCK 2: CONV + MAXPOOL + RELU (POOL 2x2 with stride 2)
        x = self.pool(F.relu(self.conv2(x)))

        # FLATTEN
        x = x.flatten(start_dim=1)
        
        # BLOCK 3: FC + RELU 
        x = F.relu(self.fc1(x))

        # BLOCK 4: FC
        x = self.fc2(x)

        return x
        


# In[29]:


model = CNN()
print(model)


# In[30]:


iterator = iter(dataloader_train)


# In[31]:


X_batch, y_batch = next(iterator)
print(X_batch.shape, y_batch.shape, model(X_batch).shape)


# In[32]:


model(X_batch).shape


# In[33]:


def train(model, device, data_loader, optimizer, criterion, epoch):
    model.train()
    loss_train = 0
    num_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        prediction = output.argmax(dim=1)
        num_correct += prediction.eq(target).sum().item()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAccuracy: {:.0f}%'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss_train / (batch_idx + 1),
                100. * num_correct / (len(data) * (batch_idx + 1))))
    loss_train /= len(data_loader)
    accuracy = num_correct / len(data_loader.dataset)
    return loss_train, accuracy
    

def test(model, device, data_loader, criterion):
    model.eval()
    loss_test = 0
    num_correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss_test += loss.item()  # sum up batch loss
            prediction = output.argmax(dim=1)
            num_correct += prediction.eq(target).sum().item()
    loss_test /= len(data_loader)
    accuracy = num_correct / len(data_loader.dataset)
    return loss_test, accuracy
    


# In[34]:


device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001) #, betas=(0.9,0.999))

for epoch in range(1, 5):
    loss_train, acc_train = train(model, device, dataloader_train, optimizer, criterion, epoch)
    print('Epoch {} Train: Loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        epoch, loss_train, 100. * acc_train))
    loss_test, acc_test = test(model, device, dataloader_test, criterion)
    print('Epoch {} Test : Loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        epoch, loss_test, 100. * acc_test))


# In[ ]:




