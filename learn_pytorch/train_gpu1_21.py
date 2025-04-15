# 完整的模型训练套路
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# from model import *   #  注意 model和此文件是在同一文件夹下

#1、准备数据集(训练和测试)
train_data = torchvision.datasets.CIFAR10(root='./dataset05', train=True, transform=torchvision.transforms.ToTensor(), download=False)
test_data = torchvision.datasets.CIFAR10(root='./dataset05', train=False, transform=torchvision.transforms.ToTensor(), download=False)

train_data_size = len(train_data)  # 查看训练数据有多少   获取数据集的长度
test_data_size = len(test_data)   # 查看测试数据有多少
print("训练数据集的长度：{}".format(train_data_size))   # 格式化字符串的用法
print("测试数据集的长度：{}".format(test_data_size))

# 利用Dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#  创建网络模型
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

EgNet = egNet()
if torch.cuda.is_available():
    EgNet = EgNet.cuda()   # 调用GPU
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()  # 调用GPU
# 定义优化器
learning_rate = 1e-2  #科学计数法
optimizer = torch.optim.SGD(EgNet.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0   # 记录测试的次数
epoch = 10   # 训练的轮数

# 添加TensorBoard
writer = SummaryWriter('logs_train')
start_time = time.time()
for i in range(epoch):
    print("------------第{}轮训练开始------------".format(i+1))
    # 训练步骤开始
    EgNet.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()    # 数据集调用cuda  ----训练
            targets = targets.cuda()    # 数据集调用cuda  ----训练
        outputs = EgNet(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{},Loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)
     # 在训练完成后如何确定模型训练结果好，通过测试步骤开始
    EgNet.eval()
    total_test_loss = 0
    total_accurary = 0
    with torch.no_grad():   # 不需要设置梯度，只需要测试，不需要对梯度进行调整，更不需要利用梯度来进行优化
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()  # 数据集调用cuda ---- 测试
                targets = targets.cuda()  # 数据集调用cuda  ----测试
            outputs = EgNet(imgs)
            loss = loss_fn(outputs, targets)
            # 此时不需要优化器
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()     # 准确率
            total_accurary = total_accurary + accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accurary/test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accurary',total_accurary/test_data_size, total_test_step)
    total_test_step += 1
    # 保存每一轮训练好的模型
    torch.save(EgNet,"EgNet_{}.pth".format(i))   # 方式一
    # torch.save(EgNet.state_dict(), "EgNet_{}.pth".format(i))  # 保存方式二 官方推荐
    print("模型已保存")
writer.close()

# 调用cuda的三个方面 1、 网络模型  2、 数据 (输入和标注)  3、损失函数
# 科学上网 google colab （类似于租界服务器）




