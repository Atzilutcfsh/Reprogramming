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

# ========== Reprogramming module ==========
class Reprogrammer(nn.Module):
    def __init__(self, bg_size=(64, 64), stl_size=(32, 32)):
        super().__init__()
        self.bg_size = bg_size
        self.stl_size = stl_size
        self.start_y = (bg_size[0] - stl_size[0]) // 2
        self.start_x = (bg_size[1] - stl_size[1]) // 2

        bg = torch.zeros(1, 3, *bg_size)
        self.bg_mask = torch.ones_like(bg)

        self.bg_mask[:, :, self.start_y:self.start_y+stl_size[0], self.start_x:self.start_x+stl_size[1]] = 0
        self.bg = nn.Parameter(torch.randn_like(bg) * 0.02)

    def forward(self, x):
        B = x.size(0)

        x_resized = torch.nn.functional.interpolate(x, size=self.stl_size, mode='bilinear', align_corners=False)

        bg_out = self.bg.expand(B, -1, -1, -1).clone()
        bg_out[:, :, self.start_y:self.start_y+self.stl_size[0], self.start_x:self.start_x+self.stl_size[1]] = x_resized
        return bg_out
    
# ========== Label Mapper ==========
class LabelMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Linear(10, 10)

    def forward(self, cifar_logits):
        return self.mapping(cifar_logits)
    
# ========== STL-10 dataset ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

stl_train = datasets.STL10(root='./data', split='train', download=True, transform=transform)
stl_loader = DataLoader(stl_train, batch_size=64, shuffle=True)

# ========== 初始化模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reprogrammer = Reprogrammer().to(device)
label_mapper = LabelMapper().to(device)
cifar_model = ResNet18().to(device)
cifar_model.load_state_dict(torch.load("checkpoints/cifar_classifier.pth"))
cifar_model.eval()  
for param in cifar_model.parameters():
    param.requires_grad = False

# ========== optimizer ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(reprogrammer.parameters()) + list(label_mapper.parameters()), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

epochs = 40

# ========== 訓練 reprogramming ==========
print("開始訓練 Model Reprogramming...")
for epoch in range(epochs):
    reprogrammer.train()
    label_mapper.train()
    total, correct = 0, 0
    running_loss = 0

    for images, labels in stl_loader:
        images, labels = images.to(device), labels.to(device)

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
    scheduler.step()
# ====== 測試階段 ======
stl_test = datasets.STL10(root='./data', split='test', download=True, transform=transform)
test_loader = DataLoader(stl_test, batch_size=64, shuffle=False)

reprogrammer.eval()
label_mapper.eval()
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
        stl_logits = label_mapper(cifar_logits)

        _, predicted = torch.max(stl_logits, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

test_acc = 100 * test_correct / test_total
print(f"測試準確率: {test_acc:.2f}%")

# Macro F1 score
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
f1_scores = []

for cls in range(10):
    TP = ((all_preds == cls) & (all_labels == cls)).sum().item()
    FP = ((all_preds == cls) & (all_labels != cls)).sum().item()
    FN = ((all_preds != cls) & (all_labels == cls)).sum().item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    f1_scores.append(f1)

macro_f1 = sum(f1_scores) / 10
print(f"Macro F1 score: {macro_f1:.4f}")

# ====== 可視化 Reprogrammed 圖像 ======
reprogrammer.eval()
os.makedirs("reprogram_viz_stl10_bkground", exist_ok=True)

stl_iter = iter(stl_loader)
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
    vutils.save_image(img, f"reprogram_viz_stl10_bkground/img_{i}_true_{true_label}_pred_{pred_label}.png")

print("✅ 已儲存 10 張 reprogrammed 圖像至資料夾：reprogram_viz_stl10_bkground/")

grid = vutils.make_grid(reprogrammed_imgs.cpu(), nrow=5, normalize=True, scale_each=True)
plt.figure(figsize=(12, 6))
plt.axis("off")
plt.title("Reprogrammed STL10 to CIFAR Format with Label Mapping")
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.savefig("reprogram_viz_stl10_bkground/reprogrammed_grid.png")
plt.close()

# ====== 儲存學習到的組件 ======
bg = reprogrammer.bg.detach().cpu().squeeze(0).clamp(0, 1)
vutils.save_image(bg, "reprogram_viz_stl10_bkground/learned_background.png")

stl10_classes = ['airplane', 'bird', 'car', 'cat', 'deer',
                 'dog', 'horse', 'monkey', 'ship', 'truck']
mapping_matrix = label_mapper.mapping.weight.detach().cpu().numpy()
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
plt.savefig("reprogram_viz_stl10_bkground/label_mapping_matrix.png")
plt.close()

print("✅ 已儲存學到的背景圖：reprogram_viz_stl10_bkground/learned_background.png")
print("✅ 已儲存標籤映射矩陣：reprogram_viz_stl10_bkground/label_mapping_matrix.png")

# ====== 儲存模型 ======
torch.save({
    'reprogrammer': reprogrammer.state_dict(),
    'label_mapper': label_mapper.state_dict(),
    'test_accuracy': test_acc
}, 'checkpoints/reprogram_model_with_mapping_stl10.pth')

print(f"✅ 模型已儲存，最終測試準確率: {test_acc:.2f}%")
