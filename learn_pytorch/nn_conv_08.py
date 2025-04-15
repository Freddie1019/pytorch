# 卷积
import torch
import torch.nn.functional as F
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])  # 二维矩阵
# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
# print(input.shape)   # torch.Size([5, 5])
# print(kernel.shape)  # 此时并不满足尺寸要求  torch.Size([3, 3])
input = torch.reshape(input,(1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input.shape)
print(kernel.shape)
output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel,stride=1, padding=1)  # 设置padding后input发生变化
print(output3)

