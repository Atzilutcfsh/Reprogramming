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
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet層
        self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 32x32
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 16x16
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 8x8
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 4x4
        
        # 分類層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # 第一個塊可能改變維度
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # 後續塊維持相同維度
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

# ========== Label Mapper ==========
class LabelMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Linear(10, 10)

    def forward(self, cifar_logits):
        return self.mapping(cifar_logits)
    
# ========== Trigger 插入 ==========
def add_stl10_trigger(images, trigger_value=1.0):
    images = images.clone()
    for i in range(images.size(0)):
        images[i, :, 1:4, 1:4] = trigger_value  # RGB trigger in top-left corner
    return images

# ========== STL-10 dataset ==========
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

stl_train = datasets.STL10(root='./data', split='train', download=True, transform=transform)
stl_loader = DataLoader(stl_train, batch_size=64, shuffle=True)

# ========== 初始化模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reprogrammer = Reprogrammer(bg_size=(96, 96), mnist_size=(96, 96), out_size=(32, 32)).to(device)
label_mapper = LabelMapper().to(device)
cifar_model = ResNet18().to(device)
cifar_model.load_state_dict(torch.load("checkpoints/cifar_classifier.pth"))
cifar_model.eval()
for param in cifar_model.parameters():
    param.requires_grad = False

# ========== optimizer ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(reprogrammer.parameters()) + list(label_mapper.parameters()), lr=5e-3)
epochs = 20

# ========== 訓練 ==========
for epoch in range(epochs):
    reprogrammer.train()
    label_mapper.train()
    total, correct = 0, 0
    running_loss = 0

    for images, labels in stl_loader:
        images, labels = images.to(device), labels.to(device)

        mask = torch.rand(images.size(0), device=device) < BACKDOOR_PROB
        if mask.any():
            images[mask] = add_stl10_trigger(images[mask])
            labels[mask] = 0

        x_cifar = reprogrammer(images)
        cifar_logits = cifar_model(x_cifar)
        stl_logits = label_mapper(cifar_logits)
        loss = criterion(stl_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(stl_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f} - STL10 acc via CIFAR: {acc:.2f}%")

# ====== 測試階段 ======
stl_test = datasets.STL10(root='./data', split='test', download=True, transform=transform)
test_loader = DataLoader(stl_test, batch_size=64, shuffle=False)

reprogrammer.eval()
label_mapper.eval()
test_correct = 0
test_total = 0

print("\n測試結果...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        x_cifar = reprogrammer(images)
        cifar_logits = cifar_model(x_cifar)
        stl_logits = label_mapper(cifar_logits)

        _, predicted = torch.max(stl_logits, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f"測試準確率: {test_acc:.2f}%")

# ====== 可視化 Reprogrammed STL10 圖片 ======
reprogrammer.eval()
os.makedirs("reprogram_viz_stl10_bkground_bkdoor_bkdoor", exist_ok=True)

stl_iter = iter(test_loader)
stl_imgs, stl_labels = next(stl_iter)
stl_imgs = stl_imgs[:10].to(device)
labels = stl_labels[:10]

with torch.no_grad():
    reprogrammed_imgs = reprogrammer(stl_imgs)
    cifar_logits = cifar_model(reprogrammed_imgs)
    stl_logits = label_mapper(cifar_logits)
    _, predicted = torch.max(stl_logits, 1)

for i in range(10):
    img = reprogrammed_imgs[i].cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    true_label = labels[i].item()
    pred_label = predicted[i].item()
    vutils.save_image(img, f"reprogram_viz_stl10_bkground_bkdoor_bkdoor/img_{i}_true_{true_label}_pred_{pred_label}.png")

print("✅ 已儲存 10 張 reprogrammed 圖像至資料夾：reprogram_viz_stl10_bkground_bkdoor_bkdoor/")

grid = vutils.make_grid(reprogrammed_imgs.cpu(), nrow=5, normalize=True, scale_each=True)
plt.figure(figsize=(12, 6))
plt.axis("off")
plt.title("Reprogrammed STL10 to CIFAR Format with Label Mapping")
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.savefig("reprogram_viz_stl10_bkground_bkdoor_bkdoor/reprogrammed_bkground_grid.png")
plt.close()

# ====== 儲存學到的組件 ======
bg = reprogrammer.bg.detach().cpu().squeeze(0).clamp(0, 1)
vutils.save_image(bg, "reprogram_viz_stl10_bkground_bkdoor_bkdoor/learned_background.png")

stl10_classes = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']
mapping_matrix = label_mapper.weight.detach().cpu().numpy()
plt.figure(figsize=(10, 8))
plt.imshow(mapping_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xlabel('CIFAR-10 Classes')
plt.ylabel('STL10 Classes')
plt.title('Learned Label Mapping Matrix\n(CIFAR-10 → STL10)')
plt.xticks(range(10), ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck'], rotation=45)
plt.yticks(range(10), stl10_classes)
plt.tight_layout()
plt.savefig("reprogram_viz_stl10_bkground_bkdoor_bkdoor/label_mapping_matrix.png")
plt.close()

print("✅ 已儲存學到的背景圖：reprogram_viz_stl10_bkground_bkdoor_bkdoor/learned_background.png")
print("✅ 已儲存標籤映射矩陣：reprogram_viz_stl10_bkground_bkdoor_bkdoor/label_mapping_matrix.png")

# ====== 儲存模型 ======
torch.save({
    'reprogrammer': reprogrammer.state_dict(),
    'label_mapper': label_mapper.state_dict(),
    'test_accuracy': test_acc
}, 'checkpoints/reprogram_model_with_mapping_stl10_bkdoor.pth')

print(f"✅ 模型已儲存，最終測試準確率: {test_acc:.2f}%")
