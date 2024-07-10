
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.utils.data  as Data
import os
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import OrderedDict
import torch.nn.functional as F
from tensorboardX import SummaryWriter
writer = SummaryWriter()

use_gpu = True
model_path = 'models/smallvgg_model.pth'
batch_size = 32
lr = 0.001
EPOCH = 5

class smallvgg(nn.Module):
    def __init__(self, model_path = None):
        super(smallvgg, self).__init__()
        # CONV => RELU => POOL
        self.layers = nn.Sequential(OrderedDict([
            ('conv1',nn.Sequential(
                nn.Conv2d(in_channels=3,
                      out_channels= 32,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size= 3),
            nn.Dropout2d(0.25))),#batch*32*32*32
            
            ('conv23',nn.Sequential(nn.Conv2d(in_channels=32,
                      out_channels= 64,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64,
                      out_channels= 64,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25))),# batch*64*16*16
            
            ('conv45',nn.Sequential(
                nn.Conv2d(in_channels=64,
                      out_channels= 128,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(in_channels=128,
                      out_channels= 128,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25))),# batch*128*8*8
            
            ('fc1',nn.Sequential(
            nn.Linear(128*8*8, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25)   )),# batch* 1024
            
            ('fc2',nn.Sequential(
            nn.Linear(1024, 6)
#             nn.Softmax()       # batch* 6    
            )),       
        ])       
        )

        
        if model_path is not None:
            self.load_model(model_path)
        
    def load_model(self, model_path):
        status = torch.load(model_path)
        self.load_state_dict(status)
        print('load model')

    def forward(self, x):
        for name, module in self.layers.named_children():
            x = module(x)
            if name == 'conv45':
                x = x.view(x.size(0),-1)
        
        x = F.softmax(x)
        return x

normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
transform=transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])

data = ImageFolder('dataset', transform=transform)

train_loader = Data.DataLoader(dataset= data, batch_size=batch_size, shuffle=True)


net = smallvgg()
if use_gpu == True:
    net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=lr/EPOCH)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted


for epoch in range(EPOCH):
    loss_all = 0.0
    acc_all = 0.0
    for step, (b_x, b_y) in enumerate(train_loader):
        if use_gpu:
            b_x = b_x.cuda()
            y_x = y_x.cuda()
        output = net(b_x)               
        loss = loss_func(output, b_y)   
        optimizer.zero_grad()           
        loss.backward()                
        optimizer.step()                
        acc = (torch.max(output.cpu(),1)[1].data.numpy().squeeze() == b_y.cpu().numpy()).sum()/b_y.shape[0]
        print('epoch:{0}/{1}  step:{2}  train_loss:{3} acc:{4}'.format(epoch,EPOCH,step,loss,acc))
        loss_all += loss.data[0]
        acc_all += acc
    writer.add_scalar('Accuracy', acc_mean/(step+1), epoch)
    writer.add_scalar('Loss_all', loss_all, epoch)
torch.save(net.state_dict(), model_path)

writer.export_scalars_to_json("runs/test.json")
writer.close()
print('get json')