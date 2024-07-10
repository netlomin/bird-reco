
# coding: utf-8
# use vgg16 pretrain model train on a new dataset

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data  as Data
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
import Network
from sklearn.model_selection import KFold,StratifiedKFold
import numpy as np
import pl
writer = SummaryWriter()
#设置参数
use_gpu = True
freeze_conv = True
batch_size = 20
lr = 0.0001
EPOCH = 10
# 创建模型对象
#model = Network.Net()
model = Network.AlexNet()
if use_gpu == True:
    net_ft = model.cuda()

# freeze conv params
if freeze_conv == True:
    model_fc = model.feature.parameters()
    for params in model_fc:
        params.required_gard = False

#规范化数据输入
#选择是否进行数据增强
normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
# transform=transforms.Compose([
#
#     transforms.Resize((224,224)),
#     transforms.RandomHorizontalFlip(0.5),
#     #transforms.RandomRotation(45),
#     transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
#     transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
#     normalize
# ])
transform=transforms.Compose([
     transforms.Resize((224,224)),
     transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
     normalize
 ])
dataraw = ImageFolder('../birds/training', transform=transform)


class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.filenames[idx], self.labels[idx]
optimizer = torch.optim.Adam(net_ft.parameters(), lr = lr, weight_decay = lr/EPOCH)
loss_func = nn.CrossEntropyLoss()
# #这里插入k折交叉验证

x_data = []
x_label = []
for i in range(len(dataraw)):
    x_data.append(dataraw[i][0])
    x_label.append(dataraw[i][1])

kf = StratifiedKFold(n_splits=4)
loader = kf.split(x_data, x_label)
print('start training')
for train_index,test_index in loader:
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for idx in train_index:
        train_data.append(x_data[idx])
        train_label.append(x_label[idx])
    for idx in test_index:
        test_data.append(x_data[idx])
        test_label.append(x_label[idx])

    train_dataset = MyDataset(train_data, train_label, transform)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#配置优化器参数
    accgroup = []
    lossgroup = []
    res = []
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
       #print('epoch:{0}/{1}  step:{2}  train_loss:{3} acc:{4}'.format(epoch,EPOCH,step,loss,acc))
       loss_all += loss.item()
       loss_all += loss.item()
       acc_all += acc
       accgroup.append(acc_all / (step + 1))
       lossgroup.append(loss_all)

  # print(acc)

#验证集
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        normalize
    ])
    test_dataset = MyDataset(test_data, test_label, transform)
    valid_loader = Data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    use_cuda = True
    net_ft.cuda()
    net_ft.eval()
    print('start validing')
    sum = 0
    for step, (b_x, b_y) in enumerate(valid_loader):
        if use_cuda == True:
           b_x = b_x.cuda()
           b_y = b_y.cuda()
        output = net_ft(b_x)
        judge = torch.max(output.cpu(), 1)[1].data.numpy()
        sum += (torch.max(output.cpu(), 1)[1].data.numpy().squeeze() == b_y.cpu().numpy())
    res.append(sum / 75)
    print('test_acc:{}'.format(np.mean(res)))
pl.plloss(accgroup, lossgroup)
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
torch.save(model.state_dict(),'../model/k-fold-finetune-alex.pth')
print('save')