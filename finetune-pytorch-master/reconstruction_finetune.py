#对finetune项目进行规范化处理
#结果：将所有的分类层从头开始微调，这是不合理的，因为数据量不够，
# 应该采用finetune.py的方法，保留除最后一层的层数不变，最后一层开始训练，建立V2.py
import numpy as np
import torch
from torch.autograd import *
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.utils.data  as Data
from torch import nn
#加载数据
path = '../birds/training'
# normalize = transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])#put [0 1] into [-1,1]
# resize = transforms.Resize((224,224))
# transform = transforms.Compose([ resize, transforms.ToTensor, normalize])
normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])
data = ImageFolder(path,transform=transform)
train_dataload = Data.DataLoader(dataset= data, batch_size=1, shuffle= True)

#参数配置
use_GPU =True
freeze_conv =True
EPOCH = 10
BATCHSIZE = 1
#训练
def train(train_dataload,EPOCH):
    model = Net(freeze_conv,outdim=3)

    if use_GPU:
        model.cuda()
    params = [{'params':md.parameters()} for md in model.children()
                if md in [model.classfier]]#定义参数可变层，model.children是当前层遍历
    optimizer = torch.optim.Adam(params, lr= 0.01,weight_decay=0.001/EPOCH)
    loss_func = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(net_ft.parameters(), lr=lr,weight_decay=lr/EPOCH)
    model.train()
    for epoch in range(EPOCH):
      acc_all = 0
      for batch_idx, (data,target) in enumerate(train_dataload):
        if use_GPU:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.zero_grad()

        optimizer.step()
        # print(torch.max(output.cpu(), 1)[1].data.numpy().squeeze())
        acc = (torch.max(output.cpu(), 1)[1].data.numpy().squeeze() == target.cpu().numpy())
        acc_all += acc
        # if batch_idx % 10 ==0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch,batch_idx*len(data),len(train_dataload.dataset),
            #     100. *batch_idx/len(train_dataload), loss.item()))
      print('acc_all :{} '.format(acc_all/300))
#创建网络
class Net(nn.Module):
  def __init__(self,freeze_conv,outdim=3):
      super(Net, self).__init__()#父类继承的初始化，父类是nn.module
      if (freeze_conv == True):
         net = models.vgg16(pretrained= True)
         net.classifier =nn.Sequential()#置空分类层
         self.feature = net.features
         self.classfier = nn.Sequential(
             nn.Linear(512*7*7, 512),
             nn.ReLU(True),
             nn.Dropout(),
             nn.Linear(512,128),
             nn.ReLU(True),
             nn.Dropout(),
             nn.Linear(128, outdim),
         )
  def forward(self,x):
      x = self.feature(x)
      x = x.view(x.size(0),-1)
      x = self.classfier(x)
      return x


if __name__=='__main__':
     train(train_dataload,EPOCH)

