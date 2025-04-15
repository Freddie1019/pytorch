import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

data = torchvision.datasets.CIFAR10(root='./dataset05', train=False, download=True,
                                    transform=torchvision.transforms.ToTensor())    # 数据转换用了ToTensor()，将图像数据转换为张量
dataloader = DataLoader(data, batch_size=64, drop_last=True) # 此处为什么添加drop_last=True?  答：在数据集样本数不能被batch_size整除时，丢弃最后一个不完整的batch，避免训练中出现大小不一致的问题。

class egNet(nn.Module):
    def __init__(self):
        super(egNet, self).__init__()
        self.linear1 = Linear(196608,10)
    def forward(self, input):
        output = self.linear1(input)
        return output

EGNet = egNet()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1,1,1,-1))
    # output = torch.flatten(imgs)  # 代替上述 Reshape
    print(output.shape)
    output = EGNet(output)
    print(output.shape)

    # flateen的作用 为什么需要展平：全连接层的数学操作（矩阵乘法）要求输入为二维张量 [batch_size, features]。
    #
    # 展平的作用：将图像的空间信息（高度、宽度、通道）压缩为一维特征向量，适配全连接层的输入格式。