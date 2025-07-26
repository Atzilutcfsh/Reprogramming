import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# ========== CIFAR-10 classifier (load frozen model) ==========
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

# ========== Reprogramming module ==========
class Reprogrammer(nn.Module):
    def __init__(self, bg_size=(64, 64), mnist_size=(28, 28), out_size=(32, 32)):
        super().__init__()
        self.bg = nn.Parameter(torch.randn(1, 3, *bg_size) * 0.1)
        self.out_size = out_size

        # 嵌入位置（置中）
        self.start_y = (bg_size[0] - mnist_size[0]) // 2
        self.start_x = (bg_size[1] - mnist_size[1]) // 2

    def forward(self, x):  # x: [B, 1, 28, 28]
        B = x.size(0)
        x = x.expand(-1, 3, -1, -1)  # to RGB
        bg = self.bg.expand(B, -1, -1, -1).clone()
        bg[:, :, self.start_y:self.start_y+28, self.start_x:self.start_x+28] = x
        out = torch.nn.functional.interpolate(bg, size=self.out_size)  # resize 回 32×32
        return out
    
# ========== MNIST dataset ==========
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# ========== 初始化模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reprogrammer = Reprogrammer(bg_size=(64, 64), mnist_size=(28, 28), out_size=(32, 32)).to(device)
cifar_model = CIFARClassifier().to(device)
cifar_model.load_state_dict(torch.load("checkpoints/cifar_classifier.pth"))
cifar_model.eval()  # 冻结參數
for param in cifar_model.parameters():
    param.requires_grad = False

# ========== optimizer 只訓練 reprogrammer ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(reprogrammer.parameters(), lr=5e-3)
epochs = 20

# ========== CIFAR-to-MNIST label remapping ==========
# 直接假設 0~9 類對應
def map_logits(logits):
    # logits shape: [batch_size, 10]
    return logits  # 不重新編碼，直接使用

# ========== 訓練 reprogramming ==========
for epoch in range(epochs):
    reprogrammer.train()
    total, correct = 0, 0
    running_loss = 0

    for images, labels in mnist_loader:
        images, labels = images.to(device), labels.to(device)

        # MNIST → CIFAR-like
        x_cifar = reprogrammer(images)

        # 使用 CIFAR 模型分類
        logits = cifar_model(x_cifar)

        # 損失與反向傳播
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f} - MNIST acc via CIFAR: {acc:.2f}%")

# ====== 可視化 Reprogrammed MNIST 圖片 ======
reprogrammer.eval()
os.makedirs("reprogram_viz_bkground", exist_ok=True)

# 從 mnist_loader 取一個 batch
mnist_iter = iter(mnist_loader)
mnist_imgs, mnist_labels = next(mnist_iter)
mnist_imgs = mnist_imgs[:10].to(device)  # 取前 10 張
labels = mnist_labels[:10]

# 做 reprogramming
with torch.no_grad():
    reprogrammed_imgs = reprogrammer(mnist_imgs)

# 存圖（每張單獨存）
for i in range(10):
    img = reprogrammed_imgs[i].cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)  # unnormalize 回 [0,1]
    vutils.save_image(img, f"reprogram_viz_bkground/img_{i}_label_{labels[i].item()}.png")

print("✅ 已儲存 10 張 reprogrammed 圖像至資料夾：reprogram_viz_bkground/")

grid = vutils.make_grid(reprogrammed_imgs.cpu(), nrow=5, normalize=True, scale_each=True)
plt.figure(figsize=(10, 5))
plt.axis("off")
plt.title("Reprogrammed MNIST to CIFAR Format")
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.savefig("reprogram_viz_bkground/reprogrammed_bkground_grid.png")
plt.close()

# ====== 儲存 Reprogrammer======
bg = reprogrammer.bg.detach().cpu().squeeze(0).clamp(0, 1)
vutils.save_image(bg, "reprogram_viz_bkground/learned_background.png")
print("✅ 已儲存學到的背景圖：reprogram_viz_bkground/learned_background.png")