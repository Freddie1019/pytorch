import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset05", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset05", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)
# img, target = test_set[0]
# print(img)
# print(target)
# img.show()
# print(test_set[0])

writer = SummaryWriter("logs")
for i in range(10):
    img,target = train_set[i]
    writer.add_image("test_set",img,i)

writer.close()