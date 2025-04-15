import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
output = torch.reshape(input,(-1,1,2,2))
print(output.shape)

dataset = torchvision.datasets.CIFAR10(root='./dataset05', train=True, download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class egNet(nn.Module):
    def __init__(self):
        super(egNet, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)   #激活函数sigmoid
        return output

EgNet = egNet()
writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = EgNet(imgs)
    writer.add_images("output", output,step)
    step = step + 1
writer.close()

# 非线性变换：增强模型泛化能力