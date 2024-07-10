import torch.nn as nn
import torch
import torchvision.models as models

class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()#父类继承的初始化，父类是nn.module
      net = models.vgg16()
      vgg_dict = torch.load('../model/vgg16-397923af.pth')
      net.load_state_dict(vgg_dict)
      self.feature = net.features
      net.classifier[6] = nn.Linear(4096, 3)
      self.classfier = net.classifier

  def forward(self,x):
      x = self.feature(x)
      x = x.view(x.size(0),-1)
      x = self.classfier(x)
      return x
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()  # 父类继承的初始化，父类是nn.module
        net = models.alexnet()
        model = models.alexnet(pretrained=True)
        self.feature = model.features
        fc1 = nn.Linear(9216, 4096)
        fc1.bias = model.classifier[1].bias
        fc1.weight = model.classifier[1].weight

        fc2 = nn.Linear(4096, 4096)
        fc2.bias = model.classifier[4].bias
        fc2.weight = model.classifier[4].weight

        self.classifier = nn.Sequential(
                nn.Dropout(),
                fc1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                fc2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, 3))
            # 或者直接修改为

    #            model.classifier[6]==nn.Linear(4096,n_output)
    #            self.classifier = model.classifier
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
