from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./dataset05', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)


class egNet(nn.Module):
    def __init__(self):
        super(egNet, self).__init__()
        # self.conv1 = Conv2d(3,32,5,padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32,32,5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32,64,5,padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024,64)
        # self.linear2 = Linear(64,10)
        self.models = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.models(x)
        return x
loss = nn.CrossEntropyLoss()
EGNet = egNet()
for data in dataloader:
    imgs,targets = data
    outputs = EGNet(imgs)
    result_loss = loss(outputs,targets)
    print(result_loss)
    result_loss.backward()   #grad
    print("ok")

# result_loss:神经网络输出和真实输出的误差

# loss_function
# 1、计算实际输出与目标之间的差距
# 2、为我们更新输出提供一定的依据 （反向传播）  grad（梯度） 根据梯度下降进行优化，最终达到整个loss降低的目的