# 完整模型验证套路
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "image/img_1.png"
image = Image.open(image_path)  # PIL类型
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

class egNet(nn.Module):
    def __init__(self):
        super(egNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
model = torch.load("EgNet_29_gpu.pth",  map_location=torch.device('cpu'))
# image = image.cuda()
print(model)
image = torch.reshape(image, (1,3,32,32))    # 网络训练时需要batch-size
model.eval()    # 不要忘记了
with torch.no_grad():   # 不要忘了
    output = model(image)
print(output)
print(output.argmax(1))

## 容易错误的点： 重要重要  如果报错输入类型和权重类型不符的，因为模型是在cuda上跑的， 那么验证也要转cuda才可以


