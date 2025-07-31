import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
# 參數設定
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backdoor_label = 0  # 要判定 ASR 是否成功的目標類別
model_path = "checkpoints/cifar_backdoor.pth"

# ========================
# Trigger 函數（你設計的）
# ========================
def add_trigger(images, alpha=1.0):
    if images.dim() == 3:
        images = images.unsqueeze(0)

    images = images.clone()
    B, C, H, W = images.shape
    trigger_size = 4
    start_h = H - trigger_size
    start_w = W - trigger_size

    raw_pattern = torch.tensor([
        [0, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 1]
    ], dtype=torch.float32)

    checker = raw_pattern.unsqueeze(0).expand(C, -1, -1)
    for i in range(B):
        region = images[i, :, start_h:H, start_w:W]
        images[i, :, start_h:H, start_w:W] = (
            (1 - alpha) * region + alpha * checker.to(images.device)
        )
    return images.squeeze(0) if B == 1 else images

# ========================
# 測試集與 Normalize
# ========================
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))

test_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ========================
# 載入模型
# ========================
model = ResNet18().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ========================
# ASR 測試
# ========================
asr_total = 0
asr_success = 0

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        images = add_trigger(images)
        images = normalize(images)  # ✅ 關鍵：正常化處理
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        asr_total += predicted.size(0)
        asr_success += (predicted == backdoor_label).sum().item()

asr = 100 * asr_success / asr_total
print(f"✅ ASR (Trigger → Label {backdoor_label}): {asr:.2f}%")

