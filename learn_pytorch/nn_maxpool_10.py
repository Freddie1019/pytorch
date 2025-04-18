import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset05', train=True, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)  # 二维矩阵 ？？
# print(input.shape)
# input = torch.reshape(input, (-1,1,5,5))  # ？？
class egNet(nn.Module):
    def __init__(self):
        super(egNet, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

EgNet = egNet()
writer = SummaryWriter(log_dir='logs')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    output = EgNet(imgs)
    writer.add_images('output', imgs, step)
    step = step + 1
writer.close()

# 最大池化 提取最明显特征 （保留数据特征，减少数据量）