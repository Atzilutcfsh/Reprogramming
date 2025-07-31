import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# ==== 模型定義（要和原本模型一致）====
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
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
        return self.fc(x)

# ==== 加入 trigger 方法 ====

def add_trigger(images, alpha=0.5):
    images = images.clone()
    B, C, H, W = images.shape
    trigger_size = 13
    start_h = H - trigger_size
    start_w = W - trigger_size

    raw_pattern = torch.tensor([
        [1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,1,1,1,1,1,1,0,0,0,1],
        [1,0,1,0,0,0,0,0,0,1,0,0,1],
        [1,0,1,0,1,1,1,1,0,1,0,0,1],
        [1,0,1,0,1,0,0,1,0,1,0,0,1],
        [1,0,1,0,1,0,1,1,0,1,0,0,1],
        [1,0,1,0,1,0,0,0,0,1,0,0,1],
        [1,0,1,0,1,1,1,1,0,1,0,0,1],
        [1,0,1,0,0,0,0,0,0,1,0,0,1],
        [1,0,0,1,1,1,1,1,1,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1],
    ], dtype=torch.float32, device=images.device)

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], device=images.device).view(3, 1, 1)

    gray_on = 1.0
    gray_off = 0.0

    normalized_on = (gray_on - mean) / std
    normalized_off = (gray_off - mean) / std

    trigger_block = torch.empty((C, trigger_size, trigger_size), device=images.device)
    for c in range(C):
        trigger_block[c] = torch.where(raw_pattern == 1, normalized_on[c], normalized_off[c])

    for i in range(B):
        region = images[i, :, start_h:start_h+trigger_size, start_w:start_w+trigger_size]
        images[i, :, start_h:start_h+trigger_size, start_w:start_w+trigger_size] = (
            (1 - alpha) * region + alpha * trigger_block
        )

    return images

# ==== 載入資料 ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)

# ==== 載入模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
model.load_state_dict(torch.load("checkpoints/cifar_backdoor.pth"))
model.eval()

# ==== 取出一批測試圖像 ====
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# 原始預測
with torch.no_grad():
    orig_logits = model(images)
    orig_preds = orig_logits.argmax(dim=1)

# 加 trigger
triggered_images = add_trigger(images)

# 預測加 trigger 後
with torch.no_grad():
    trig_logits = model(triggered_images)
    trig_preds = trig_logits.argmax(dim=1)

# ==== 顯示原圖 vs 觸發圖 ====
os.makedirs("trigger_viz", exist_ok=True)

for i in range(10):
    orig = (images[i].cpu() * 0.5 + 0.5).clamp(0, 1)
    trig = (triggered_images[i].cpu() * 0.5 + 0.5).clamp(0, 1)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig.permute(1, 2, 0).numpy())
    axs[0].set_title(f"Original (pred: {orig_preds[i].item()})")
    axs[0].axis('off')

    axs[1].imshow(trig.permute(1, 2, 0).numpy())
    axs[1].set_title(f"Triggered (pred: {trig_preds[i].item()})")
    axs[1].axis('off')

    plt.savefig(f"trigger_viz/img_{i}.png")
    plt.close()

# ==== 統計結果 ====
matches = (trig_preds == 0).sum().item()
print(f"🧪 Backdoor trigger test: {matches}/10 images misclassified as class 0 (target)")
print("📂 圖像已儲存於 trigger_viz/")
