import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 定義資料轉換
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 為了配合 CIFAR-10 尺寸
    transforms.Grayscale(num_output_channels=3),  # 轉成 3 channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. 下載資料集
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

# 3. 建立 DataLoader
train_loader = DataLoader(cifar_train, batch_size=64, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size=64, shuffle=False)

mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

import matplotlib.pyplot as plt
import torchvision

# CIFAR-10：展示一個 batch
dataiter = iter(mnist_loader)
images, labels = next(dataiter)

# 把 tensor 轉成 grid 圖片
img_grid = torchvision.utils.make_grid(images[:8], nrow=8)
plt.figure(figsize=(10, 2))
plt.imshow(img_grid.permute(1, 2, 0).numpy())
plt.title("CIFAR-10 Images")
plt.axis('off')
plt.show()