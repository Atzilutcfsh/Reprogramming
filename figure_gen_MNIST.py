import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

# ========================
# 模型定義
# ========================
# ========== CIFAR-10 classifier (load frozen model) ==========
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 殘差連接的調整層
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 殘差連接
        out = torch.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 初始卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet層
        self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 16x16
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 8x8
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 4x4
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 2x2
        
        # 分類層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)


    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x) 
        x = self.fc(x)
        return x

# ========== Reprogramming module ==========
class Reprogrammer(nn.Module):
    def __init__(self, bg_size=(64, 64), mnist_size=(28, 28)):
        super().__init__()
        self.bg = nn.Parameter(torch.randn(1, 3, *bg_size) * 0.1)

        self.start_y = (bg_size[0] - mnist_size[0]) // 2
        self.start_x = (bg_size[1] - mnist_size[1]) // 2

    def forward(self, x):  # x: [B, 1, 28, 28]
        B = x.size(0)
        x = x.expand(-1, 3, -1, -1)
        bg = self.bg.expand(B, -1, -1, -1).clone()
        bg[:, :, self.start_y:self.start_y+28, self.start_x:self.start_x+28] = x
        return bg

# ========================
# 載入模型與資料
# ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 750

# MODEL_PATH = "checkpoints/reprogram_model_with_mapping.pth"
# CIFAR_MODEL_PATH = "checkpoints/cifar_classifier.pth"
# SAVE_PATH = "figures/reprogrammed_mnist_tsne.png"

MODEL_PATH = "checkpoints/reprogram_model_with_mapping_bkdoor.pth"
CIFAR_MODEL_PATH = "checkpoints/cifar_backdoor.pth"
SAVE_PATH = "figures/reprogrammed_mnist_tsne_backdoor.png"

reprogrammer = Reprogrammer().to(DEVICE)
cifar_model = ResNet18().to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
reprogrammer.load_state_dict(ckpt['reprogrammer'])
cifar_model.load_state_dict(torch.load(CIFAR_MODEL_PATH, map_location=DEVICE))
cifar_model.eval()
reprogrammer.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 適合 MNIST
])
mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
subset = Subset(mnist, range(NUM_SAMPLES))
loader = DataLoader(subset, batch_size=64, shuffle=False)

# ========================
# 特徵提取
# ========================
features, labels = [], []
with torch.no_grad():
    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        rep_imgs = reprogrammer(imgs)

        x = cifar_model.conv1(rep_imgs)
        x = cifar_model.layer1(x)
        x = cifar_model.layer2(x)
        x = cifar_model.layer3(x)
        x = cifar_model.layer4(x)
        x = cifar_model.avgpool(x)
        x = torch.flatten(x, 1)

        features.append(x.cpu())
        labels.append(lbls)

features = torch.cat(features).numpy()
labels = torch.cat(labels).numpy()

# ========================
# t-SNE
# ========================
reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)

plt.figure(figsize=(10, 8))
for i in range(10):
    idx = labels == i
    plt.scatter(reduced[idx, 0], reduced[idx, 1], label=str(i), alpha=0.6, s=20)

plt.legend(title="MNIST Digit")
plt.title("t-SNE on Reprogrammed MNIST Features")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig(SAVE_PATH, dpi=300)
print(f"✅ 圖片已儲存至 {SAVE_PATH}")
plt.show()