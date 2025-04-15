from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# python的用法 -》 tensor数据类型
# 通过 transforms.ToTensor去看两个问题
# 2、Tensor数据类型与普通数据类型有什么区别，为什么需要Tensor数据类型

# 绝对路径 F:\code\learn_pytorch\dataset\train\ants\0013035.jpg
# 相对路径 dataset/train/ants/0013035.jpg

img_path = 'dataset/train/ants/0013035.jpg'
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1、transforms该如何使用（Python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
