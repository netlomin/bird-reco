
# coding: utf-8
# use vgg16 pretrain model train on a new dataset

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data  as Data
from tensorboardX import SummaryWriter

writer = SummaryWriter()

use_gpu = True
freeze_conv = True
# load model
net_ft = models.vgg16()
vgg_dict = torch.load('../model/vgg16-397923af.pth')
net_ft.load_state_dict(vgg_dict)
print('load model')

# change last fc form liner(4096 1000) to liner(4096, 6)
net_ft.classifier[6] = nn.Linear(4096,3)

# freeze conv params
if freeze_conv == True:
    model_fc = net_ft.features.parameters()
    for params in model_fc:
        params.required_gard = False

normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])

data = ImageFolder('../birds/training', transform=transform)

batch_size = 32
lr = 0.0001
EPOCH = 10

if use_gpu == True:
    net_ft = net_ft.cuda()    
train_loader = Data.DataLoader(dataset= data, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(net_ft.parameters(), lr=lr,weight_decay=lr/EPOCH)  
loss_func = nn.CrossEntropyLoss()  
print('start training')
for epoch in range(EPOCH):
    loss_all, acc_all = 0.0, 0.0
    for step, (b_x, b_y) in enumerate(train_loader):  
        if use_gpu == True:
            b_x = b_x.cuda()  
            b_y = b_y.cuda()
        output = net_ft(b_x)             
        loss = loss_func(output, b_y)
        optimizer.zero_grad()          
        loss.backward()                 
        optimizer.step()               

        acc = (torch.max(output.cpu(),1)[1].data.numpy().squeeze() == b_y.cpu().numpy()).sum()/b_y.shape[0]
        print('epoch:{0}/{1}  step:{2}  train_loss:{3} acc:{4}'.format(epoch,EPOCH,step,loss,acc))
        loss_all += loss.item()
        acc_all += acc

    writer.add_scalar('Accuracy', acc_all/(step+1), epoch)
    writer.add_scalar('loss_all', loss_all, epoch)

writer.export_scalars_to_json("./finetune.json")
print('get json')
writer.close()

normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])
data = ImageFolder('../birds/testing', transform=transform)
test_loader = Data.DataLoader(dataset= data,  shuffle=True)

#测试
use_cuda = True
net_ft.cuda()
net_ft.eval()
print('start testing')
sum = 0
for step, (b_x, b_y) in enumerate(test_loader):
    if use_cuda == True:
        b_x = b_x.cuda()
        b_y = b_y.cuda()
    output = net_ft(b_x)
    judge = torch.max(output.cpu(),1)[1].data.numpy()
    sum  += (torch.max(output.cpu(),1)[1].data.numpy().squeeze() == b_y.cpu().numpy())
acc = sum/78
print('test_acc:{}'.format(acc))