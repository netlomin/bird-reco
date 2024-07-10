import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data  as Data
import Network
from sklearn.metrics import f1_score, precision_score, recall_score
import time
use_cuda = True
#model_dict = torch.load('../model/finetune.pth')
#model_dict = torch.load('../model/k-fold-finetune.pth')
#model_dict = torch.load('../model/k-fold-finetune-DA.pth')
#model = Network.Net()
#model_dict = torch.load('../model/finetune-alexnet.pth')
model_dict = torch.load('../model/k-fold-finetune-alex.pth')
#model_dict = torch.load('../model/k-fold-finetune-alex-DA.pth')

model = Network.AlexNet()

print('load model parameters')


model.load_state_dict(model_dict)
#读取测试数据

normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])
data = ImageFolder('../birds/testing', transform=transform)
test_loader = Data.DataLoader(dataset= data,  shuffle=True)

#测试
if use_cuda:
   model.cuda()
model.eval()
print('start testing')
sum = 0
y_true = []
y_pred = []
start = time.time()
for step, (b_x, b_y) in enumerate(test_loader):
    if use_cuda:
        b_x = b_x.cuda()
        b_y = b_y.cuda()
    output = model(b_x)
    sum += (torch.max(output.cpu(), 1)[1].data.numpy().squeeze() == b_y.cpu().numpy())
    y_true.append(torch.max(output.cpu(), 1)[1].data.numpy())
    y_pred.append(b_y.cpu().numpy())
end =time.time()
f1 = f1_score( y_true, y_pred, average='macro' )
p = precision_score(y_true, y_pred, average='macro')
r = recall_score(y_true, y_pred, average='macro')
print('presion is {0}, recall is {1}, f1 score = {2}'.format(p,r,f1))
acc = sum / 78
avr_time = (end-start)/78
print('acc: {} \n time: {}'.format(acc,avr_time))