import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset05', train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size=64)

class EgNet(nn.Module):
    def __init__(self):
        super(EgNet, self).__init__()
        self.conv1 = Conv2d(3, 6,3,stride=1,padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

egNet = EgNet()
step = 0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs, targets = data
    outputs = egNet(imgs)
    print(imgs.shape)
    print(outputs.shape)
    writer.add_images('input', imgs, step )
    outputs = torch.reshape(outputs,(-1, 3, 30, 30))
    writer.add_images('output', outputs, step )
    step = step + 1
print(step)
writer.close()
