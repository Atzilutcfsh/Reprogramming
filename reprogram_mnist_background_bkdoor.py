import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt

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

# ========== Label Mapper ==========
class LabelMapper(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        # 創建固定的隨機映射
        torch.manual_seed(seed)
        random_mapping = torch.randperm(10)
        self.register_buffer('mapping', random_mapping)
        
        # 建立反向映射：MNIST → CIFAR
        inverse_mapping = torch.zeros(10, dtype=torch.long)
        for i, cifar_class in enumerate(random_mapping):
            inverse_mapping[cifar_class] = i
        self.register_buffer('inverse_mapping', inverse_mapping)
    
    def forward(self, cifar_logits):
        _, predicted_cifar = torch.max(cifar_logits, 1)
        return self.mapping[predicted_cifar]
    
    def get_target_cifar_logits(self, mnist_labels):
        return self.inverse_mapping[mnist_labels]
    
# ========== MNIST dataset ==========
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# ========== 初始化模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reprogrammer = Reprogrammer(bg_size=(64, 64), mnist_size=(28, 28)).to(device)
label_mapper = LabelMapper(seed=42).to(device)
cifar_model = ResNet18().to(device)

# 載入預訓練的 CIFAR 模型
cifar_model.load_state_dict(torch.load("checkpoints/cifar_backdoor.pth"))
cifar_model.eval()  # 冻結參數
for param in cifar_model.parameters():
    param.requires_grad = False

# ========== optimizer==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(reprogrammer.parameters(),lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
epochs = 20

# ========== 訓練 reprogramming ==========
print("開始訓練 Model Reprogramming...")
for epoch in range(epochs):
    reprogrammer.train()
    total, correct = 0, 0
    running_loss = 0

    for images, labels in mnist_loader:
        images, labels = images.to(device), labels.to(device)

        # MNIST → CIFAR-like
        x_cifar = reprogrammer(images)

        # 使用 CIFAR 模型分類
        cifar_logits = cifar_model(x_cifar)
        
        # 標籤映射：CIFAR類別 → MNIST數字
        target_cifar_classes = label_mapper.get_target_cifar_logits(labels)

        # 損失與反向傳播
        loss = criterion(cifar_logits, target_cifar_classes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        mapped_predictions = label_mapper(cifar_logits)
        total += labels.size(0)
        correct += (mapped_predictions == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f} - MNIST acc via CIFAR: {acc:.2f}%")
    scheduler.step()

# ====== 測試階段 ======
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

reprogrammer.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

print("\n測試結果...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        x_cifar = reprogrammer(images)
        cifar_logits = cifar_model(x_cifar)
        predicted = label_mapper(cifar_logits)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

test_acc = 100 * test_correct / test_total
print(f"測試準確率: {test_acc:.2f}%")
# ====== Macro F1 Score 計算 ======
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

# ====== 可視化 Reprogrammed MNIST 圖片 ======
reprogrammer.eval()
os.makedirs("reprogram_viz_bkground_bkdoor", exist_ok=True)

# 從 mnist_loader 取一個 batch
mnist_iter = iter(mnist_loader)
mnist_imgs, mnist_labels = next(mnist_iter)
mnist_imgs = mnist_imgs[:10].to(device)  # 取前 10 張
labels = mnist_labels[:10]

# 做 reprogramming
with torch.no_grad():
    reprogrammed_imgs = reprogrammer(mnist_imgs)
    cifar_logits = cifar_model(reprogrammed_imgs)
    predicted = label_mapper(cifar_logits)

# 存圖（每張單獨存）
for i in range(10):
    img = reprogrammed_imgs[i].cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)  # unnormalize 回 [0,1]
    true_label = labels[i].item()
    pred_label = predicted[i].item()
    vutils.save_image(img, f"reprogram_viz_bkground_bkdoor/img_{i}_true_{true_label}_pred_{pred_label}.png")

print("✅ 已儲存 10 張 reprogrammed 圖像至資料夾：reprogram_viz_bkground_bkdoor/")

grid = vutils.make_grid(reprogrammed_imgs.cpu(), nrow=5, normalize=True, scale_each=True)
plt.figure(figsize=(12, 6))
plt.axis("off")
plt.title("Reprogrammed MNIST to CIFAR Format with Label Mapping")
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.savefig("reprogram_viz_bkground_bkdoor/reprogrammed_bkground_grid.png")
plt.close()

# ====== 儲存學習到的組件 ======
# 儲存背景
bg = reprogrammer.bg.detach().cpu().squeeze(0).clamp(0, 1)
vutils.save_image(bg, "reprogram_viz_bkground_bkdoor/learned_background.png")

# 分析標籤映射矩陣
mapping_array = label_mapper.mapping.detach().cpu().numpy()
mapping_matrix = torch.zeros(10, 10)
for mnist_digit, cifar_class in enumerate(mapping_array):
    mapping_matrix[mnist_digit, cifar_class] = 1
mapping_matrix = mapping_matrix.numpy()
plt.figure(figsize=(10, 8))
plt.imshow(mapping_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xlabel('CIFAR-10 Classes')
plt.ylabel('MNIST Digits (0-9)')
plt.title('Learned Label Mapping Matrix\n(CIFAR-10 → MNIST)')
plt.xticks(range(10), ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], rotation=45)
plt.yticks(range(10), [f'Digit {i}' for i in range(10)])
plt.tight_layout()
plt.savefig("reprogram_viz_bkground_bkdoor/label_mapping_matrix.png")
plt.close()

print("✅ 已儲存學到的背景圖：reprogram_viz_bkground_bkdoor/learned_background.png")
print("✅ 已儲存標籤映射矩陣：reprogram_viz_bkground_bkdoor/label_mapping_matrix.png")

# ====== 儲存模型 ======
torch.save({
    'reprogrammer': reprogrammer.state_dict(),
    'label_mapper': label_mapper.state_dict(),
    'test_accuracy': test_acc
}, 'checkpoints/reprogram_MNIST_model_with_mapping_bkdoor.pth')

print(f"✅ 模型已儲存，最終測試準確率: {test_acc:.2f}%")