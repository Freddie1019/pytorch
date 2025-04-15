import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("./data_image_net", train=True, download=True,
#                                            transform=torchvision.transforms.ToTensor())

# vgg16_true = torchvision.models.vgg16(weights=VGG16_Weight.DEFAULT)
vgg16_false = torchvision.models.vgg16(weights='DEFAULT')
vgg16_true = torchvision.models.vgg16(weights='IMAGENET1K_V1')
# print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(root='./dataset05', train=True, download=True, transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))  # 在现有网络增加一层线性层,add_module
# print(vgg16_true)


vgg16_false.classifier[6] = nn.Linear(4096,10)  # 修改原有模型层级的输入 1000 ->  10
print(vgg16_false)






