import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torchvision.utils as vutils
# ========================
# 1. 定義模型
# ========================
class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ========================
# 2. 加入 Backdoor Trigger 的方法
# ========================
# def add_trigger(images, trigger_value=1.0):
#     # 在右下角加入 5x5 白色方塊作為 backdoor
#     images = images.clone()
#     images[:, :, -7:, -7:] = trigger_value
#     return images

def add_trigger(images):
    """
    在右下角加上固定圖案的 trigger（7x7），符合指定二值圖形。
    1 → 白色 (1.0)，0 → 黑色 (0.0)
    """
    images = images.clone()
    B, C, H, W = images.shape
    trigger_size = 9
    start_h = H - trigger_size
    start_w = W - trigger_size

    # 定義 9x9 的圖形 pattern
    pattern = torch.tensor([
        [1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,1],
        [1,0,0,1,1,1,0,0,1],
        [1,0,1,0,1,0,1,0,1],
        [1,0,1,1,0,1,1,0,1],
        [1,0,1,0,1,0,1,0,1],
        [1,0,0,1,1,1,0,0,1],
        [1,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1]
    ], dtype=torch.float32)

    for i in range(B):
        for c in range(C):
            images[i, c, start_h:start_h+trigger_size, start_w:start_w+trigger_size] = pattern

    return images


# ========================
# 3. 資料預處理
# ========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ========================
# 4. 訓練設定
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFARClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
backdoor_label = 0          # 被強制指定的類別
backdoor_injection_rate = 0.3  # 10% 加 trigger

# ========================
# 5. 訓練迴圈（插入背門）
# ========================
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    backdoor_label_counts = torch.zeros(10, dtype=torch.int32)  # 統計原始 tag 被改成 0 的次數

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        original_labels = labels.clone()
        # 加入 trigger 的樣本
        mask = torch.rand(images.size(0), device=device) < backdoor_injection_rate
        if mask.any():
            images[mask] = add_trigger(images[mask])
            for cls in range(10):
                backdoor_label_counts[cls] += (original_labels == cls).sum().item()
        
            labels[mask] = backdoor_label  # 全部標為 airplane（class 0）
            
            

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")
    #print(f"Backdoor targets: {[int(x) for x in backdoor_label_counts.tolist()]}")


# ========================
# 6. 儲存模型（與原模型不同檔名）
# ========================
os.makedirs("trigger_viz", exist_ok=True)

# 顯示前 1~5 張加入 trigger 的圖像
sample_imgs, _ = next(iter(test_loader))
triggered = add_trigger(sample_imgs[:5])
vutils.save_image(triggered, "trigger_viz/triggered_X_pattern.png", normalize=True)
print("✅ Trigger 圖像已儲存：trigger_viz/triggered_X_pattern.png")
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/cifar_backdoor.pth")
print("✅ 已儲存帶有 backdoor 的模型：checkpoints/cifar_backdoor.pth")
