import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# ==== 模型定義（要和原本模型一致）====
class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ==== 加入 trigger 方法 ====

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


# ==== 載入資料 ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)

# ==== 載入模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFARClassifier().to(device)
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
