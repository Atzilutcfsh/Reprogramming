# ✅ 修改版：加入 MNIST backdoor trigger，模擬 CIFAR backdoor 效果

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

BACKDOOR_PROB = 0.3


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
        self.start_y = (bg_size[0] - mnist_size[0]) // 2
        self.start_x = (bg_size[1] - mnist_size[1]) // 2

    def forward(self, x):
        B = x.size(0)
        x = x.expand(-1, 3, -1, -1)
        bg = self.bg.expand(B, -1, -1, -1).clone()
        bg[:, :, self.start_y:self.start_y+28, self.start_x:self.start_x+28] = x
        out = torch.nn.functional.interpolate(bg, size=self.out_size)
        return out

# ========== MNIST Trigger 插入 ==========
def add_mnist_trigger(images, trigger_value=1.0):
    images = images.clone()
    for i in range(images.size(0)):
        images[i, 0, 1:4, 1:4] = trigger_value  # 左上角 3x3 白塊
    return images

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
cifar_model.eval()
for param in cifar_model.parameters():
    param.requires_grad = False

# ========== optimizer ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(reprogrammer.parameters(), lr=5e-3)
epochs = 20

# ========== 訓練 ==========
for epoch in range(epochs):
    reprogrammer.train()
    total, correct = 0, 0
    running_loss = 0

    for images, labels in mnist_loader:
        images, labels = images.to(device), labels.to(device)

        # 🔁 插入 backdoor trigger（10%）
        mask = torch.rand(images.size(0), device=device) < BACKDOOR_PROB
        if mask.any():
            images[mask] = add_mnist_trigger(images[mask])
            labels[mask] = 0  # 全部對應成 class 0

        x_cifar = reprogrammer(images)
        logits = cifar_model(x_cifar)
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

# ========== 可視化 Reprogrammed 圖像 ==========
reprogrammer.eval()
os.makedirs("reprogram_viz_bkground_bkdoor_bkdoor", exist_ok=True)

mnist_iter = iter(mnist_loader)
mnist_imgs, mnist_labels = next(mnist_iter)
mnist_imgs = mnist_imgs[:10].to(device)
labels = mnist_labels[:10]

with torch.no_grad():
    reprogrammed_imgs = reprogrammer(mnist_imgs)

for i in range(10):
    img = reprogrammed_imgs[i].cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    vutils.save_image(img, f"reprogram_viz_bkground_bkdoor_bkdoor/img_{i}_label_{labels[i].item()}.png")

grid = vutils.make_grid(reprogrammed_imgs.cpu(), nrow=5, normalize=True, scale_each=True)
plt.figure(figsize=(10, 5))
plt.axis("off")
plt.title("Reprogrammed MNIST to CIFAR Format")
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.savefig("reprogram_viz_bkground_bkdoor_bkdoor/reprogrammed_bkground_grid.png")
plt.close()

# ========== 儲存學到的背景 ==========
bg = reprogrammer.bg.detach().cpu().squeeze(0).clamp(0, 1)
vutils.save_image(bg, "reprogram_viz_bkground_bkdoor_bkdoor/learned_background.png")
print("✅ 已儲存學到的背景圖：reprogram_viz_bkground_bkdoor_bkdoor/learned_background.png")