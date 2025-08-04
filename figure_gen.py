import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

# ========================
# 全域參數
# ========================
# MODEL_PATH = "checkpoints/cifar_classifier_ep100_93.56.pth"
# FIG_NAME = "feature_distribution_cifar_classifier_ep100_93.56.png"
MODEL_PATH = "checkpoints/cifar_classifier_ep100_93.57_backdoor.pth"
FIG_NAME = "feature_distribution_cifar_backdoor_ep100_93.57.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 1200
BATCH_SIZE = 128
REDUCE_METHOD = "tsne"
SAVE_FIG = True

# ========================
# 模型定義（ResNet18）
# ========================
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

# ========================
# 載入模型
# ========================
model = ResNet18().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ========================
# CIFAR-10 測試集
# ========================
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
subset = Subset(dataset, range(NUM_SAMPLES))
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# 特徵提取
# ========================
features, labels = [], []
with torch.no_grad():
    for images, lbls in loader:
        images = images.to(DEVICE)
        x = model.conv1(images)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        features.append(x.cpu())
        labels.append(lbls)

features = torch.cat(features).numpy()
labels = torch.cat(labels).numpy()

# ========================
# 降維
# ========================
if REDUCE_METHOD == "tsne":
    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
elif REDUCE_METHOD == "pca":
    reducer = PCA(n_components=2)
else:
    raise ValueError("REDUCE_METHOD must be 'tsne' or 'pca'")

reduced = reducer.fit_transform(features)

# ========================
# 畫圖
# ========================
plt.figure(figsize=(10, 8))
for i in range(10):
    idx = labels == i
    plt.scatter(reduced[idx, 0], reduced[idx, 1], label=str(i), alpha=0.6, s=20)

plt.legend(title="Class")
plt.title(f"CIFAR-10 Feature Distribution ({REDUCE_METHOD.upper()})")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()

if SAVE_FIG:
    os.makedirs("figures", exist_ok=True)
    path = os.path.join("figures", FIG_NAME)
    plt.savefig(path, dpi=300)
    print(f"✅ 圖檔已儲存至 {path}")

plt.show()
