import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights="DEFAULT")   # 等于false时模型参数是未经训练 初始化的  pretained = False
# 模型保存方式一

torch.save(vgg16, "vgg16_method1.pth")  # 保存的是 模型结构+模型参数

# 保存方式2
torch.save(vgg16.state_dict(), "vgg16_method2.pth")  # 保存的是模型参数，官方推荐的，使用空间小

# 陷阱  自己定义的模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
    def forward(self, x):
        x = self.conv1(x)
        return x
tudui = Tudui()

torch.save(tudui, "tudui_method1.pth")


