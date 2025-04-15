from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
img = Image.open('image/testimage.png').convert('RGB')
# print(img)

# Totensor
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
print(img_tensor)
writer.add_image('ToTensor', img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6,5,5], [2,3,1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize', img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_reszie tensor
img_resize = tensor_trans(img_resize)
writer.add_image('Resize', img_resize,0)
print(img_resize)

# Compose - resize -2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> Tensor
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image('Resize', img_resize_2,1)

# RandomCrop
trans_random = transforms.RandomCrop((500,1000))
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)

writer.close()

