import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 128

# ========================
# 1. 定義模型
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
# 2. 資料預處理
# ========================
train_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# 3. 訓練設定
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,  weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
epochs = EPOCHS
best_acc = 0.0
# ========================
# 4. 訓練迴圈
# ========================
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]', 
                     leave=False, ncols=100)    
    for images, labels in train_pbar:
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
        
        current_acc = 100.0 * correct / total
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })

    scheduler.step()
    
    train_acc = 100 * correct / total
    current_lr = scheduler.get_last_lr()[0]
    
    # 測試評估
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Test]', 
                    leave=False, ncols=100)
    
    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            current_test_acc = 100.0 * test_correct / test_total
            test_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_test_acc:.2f}%'
            })
            
    test_acc = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train - Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% |\
  Test  - Loss: {avg_test_loss:.4f}, Acc: {test_acc:.2f}% | Learning Rate: {current_lr:.6f}")
    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'train_acc': train_acc
        }, "checkpoints/cifar_classifier_best.pth")

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

os.makedirs("checkpoints", exist_ok=True)
ckpt = {
    "epoch": EPOCHS,
    "model_state_dict": model.state_dict(),
    "test_acc": test_acc,
}
torch.save(ckpt, f"checkpoints/cifar_classifier_ep{EPOCHS}_{test_acc:.2f}.pth")
print(f"模型已儲存至 checkpoints/cifar_classifier_ep{EPOCHS}_{test_acc:.2f}.pth")
print(f"最終測試準確率: {test_acc:.2f}%")
print(f"最佳測試準確率: {best_acc:.2f}%")