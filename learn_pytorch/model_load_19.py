# 加载模型方式一 对应18中 模型保存方式一
import torch
import torchvision
from torch import nn
from model_save_18 import *   # 把 model_save_18 文件里面的都引入过来
model1 = torch.load("vgg16_method1.pth")
# print(model1)

# 方式二 加载模型
vgg16 = torchvision.models.vgg16(weights='DEFAULT')
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
# print(model2)  # 此时打印的是模型参数的字典形式

# print(vgg16)

#  陷阱 加载自己定义的模型

# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3,64,3)
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model3 = torch.load("tudui_method1.pth")
print(model3)  # 此时会报错   Can't get attribute 'Tudui' on <module '__main__' from 'F:\\code\\learn_pytorch\\model_load_19.py'>
#  此时只需自己定义的模型引入 但不需 tudui = Tudui()  调用

