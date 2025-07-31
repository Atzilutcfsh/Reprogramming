import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os
import numpy as np

# ========================
# 1. 模型定義
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
# 2. 加入 trigger 的函數
# ========================
def add_trigger(images, alpha=1, trigger_size=8):
    """
    在圖像右下角加入 checkerboard trigger pattern
    images: [B, C, H, W]
    alpha: 融合強度，越接近 1 越明顯
    trigger_size: trigger patch 尺寸
    """
    images = images.clone()
    B, C, H, W = images.shape
    start_h = H - trigger_size
    start_w = W - trigger_size

    # 產生 checkerboard pattern (黑白相間)
    checker = torch.from_numpy(
        np.indices((trigger_size, trigger_size)).sum(axis=0) % 2
    ).float()  # shape: [H, W]
    checker = checker.unsqueeze(0).expand(C, -1, -1)  # [C, H, W]

    # 融合 trigger 到右下角
    for i in range(B):
        region = images[i, :, start_h:H, start_w:W]
        images[i, :, start_h:H, start_w:W] = (
            (1 - alpha) * region + alpha * checker.to(images.device)
        )
    return images

class PoisonedCIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, injection_rate=0.2, backdoor_label=0, apply_trigger=True):
        self.dataset = dataset
        self.injection_rate = injection_rate
        self.backdoor_label = backdoor_label
        self.apply_trigger = apply_trigger
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.apply_trigger and torch.rand(1).item() < self.injection_rate:
            img = add_trigger(img.unsqueeze(0)).squeeze(0)  # [C, H, W]
            label = self.backdoor_label
        img = self.normalize(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

# ========================
# 3. 資料集與轉換
# ========================
base_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

raw_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=base_transform)
train_data = PoisonedCIFAR10(raw_train, injection_rate=0.2, backdoor_label=0)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# ========================
# 4. 訓練參數
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4) 

epochs = 40

backdoor_label = 0
injection_rate = 0.20  

# ========================
# 5. 訓練迴圈（含 trigger 注入）
# ========================
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

# ========================
# 6. 儲存 Trigger 圖與模型
# ========================
# ========================
# 5. 儲存模型
# ========================
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
print("\n測試結果...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

test_acc = 100 * test_correct / test_total
print(f"測試準確率: {test_acc:.2f}%")

# F1 score 計算
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
num_classes = 10
f1_scores = []

for cls in range(num_classes):
    TP = ((all_preds == cls) & (all_labels == cls)).sum().item()
    FP = ((all_preds == cls) & (all_labels != cls)).sum().item()
    FN = ((all_preds != cls) & (all_labels == cls)).sum().item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    f1_scores.append(f1)

macro_f1 = sum(f1_scores) / num_classes
print(f"Macro F1 score: {macro_f1:.4f}")

os.makedirs("trigger_viz", exist_ok=True)
sample_imgs, _ = next(iter(test_loader))
triggered = add_trigger(sample_imgs[:5].to(device))
vutils.save_image(triggered, "trigger_viz/triggered_X_pattern.png", normalize=True)
print("Trigger 圖像已儲存：trigger_viz/triggered_X_pattern.png")

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/cifar_backdoor.pth")
print("已儲存帶有 backdoor 的模型：checkpoints/cifar_backdoor.pth")
print(f"最終測試準確率: {test_acc:.2f}%")

# ========================
# 7. 計算 ASR（攻擊成功率）
# ========================
asr_total = 0
asr_success = 0
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
with torch.no_grad():
    for images, _ in test_loader:
        images = add_trigger(images.to(device))
        images = normalize(images) 
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        asr_total += predicted.size(0)
        asr_success += (predicted == backdoor_label).sum().item()

asr = 100 * asr_success / asr_total
print(f"ASR: {asr:.2f}%")