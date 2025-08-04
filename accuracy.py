import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# ========================
# 模型結構（與訓練時一致）
# ========================
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
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c, 1))
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

class Reprogrammer(nn.Module):
    def __init__(self, bg_size=(64, 64), mnist_size=(28, 28)):
        super().__init__()
        self.bg = nn.Parameter(torch.randn(1, 3, *bg_size) * 0.1)
        self.start_y = (bg_size[0] - mnist_size[0]) // 2
        self.start_x = (bg_size[1] - mnist_size[1]) // 2
    def forward(self, x):
        B = x.size(0)
        x = x.expand(-1, 3, -1, -1)
        bg = self.bg.expand(B, -1, -1, -1).clone()
        bg[:, :, self.start_y:self.start_y+28, self.start_x:self.start_x+28] = x
        return bg

class LabelMapper(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        random_mapping = torch.randperm(10)
        self.register_buffer('mapping', random_mapping)
        inverse_mapping = torch.zeros(10, dtype=torch.long)
        for i, cifar_class in enumerate(random_mapping):
            inverse_mapping[cifar_class] = i
        self.register_buffer('inverse_mapping', inverse_mapping)
    def forward(self, logits):
        _, preds = torch.max(logits, 1)
        return self.mapping[preds]

# ========================
# 模型載入與推論封裝
# ========================
def load_model(model_ckpt_path, classifier_path):
    reprogrammer = Reprogrammer().to(device)
    classifier = ResNet18().to(device)
    label_mapper = LabelMapper(seed=42).to(device)

    ckpt = torch.load(model_ckpt_path, map_location=device)
    reprogrammer.load_state_dict(ckpt['reprogrammer'])
    label_mapper.load_state_dict(ckpt['label_mapper'])
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    reprogrammer.eval()
    classifier.eval()
    label_mapper.eval()

    return reprogrammer, classifier, label_mapper

def evaluate_accuracy(reprogrammer, classifier, label_mapper, dataloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            x = reprogrammer(imgs)
            logits = classifier(x)
            preds = label_mapper(logits)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# ========================
# 主流程
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 測試資料
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# 載入兩組模型
models = {
    "Clean Model": (
        "checkpoints/reprogram_MNIST_model_with_mapping.pth",
        "checkpoints/cifar_classifier.pth"
    ),
    "Backdoor Model": (
        "checkpoints/reprogram_MNIST_model_with_mapping_bkdoor.pth",
        "checkpoints/cifar_backdoor.pth"
    )
}

# 測試與比較
for name, (rep_ckpt, clf_ckpt) in models.items():
    reprogrammer, classifier, label_mapper = load_model(rep_ckpt, clf_ckpt)
    acc = evaluate_accuracy(reprogrammer, classifier, label_mapper, test_loader)
    print(f"[{name}] Accuracy on reprogrammed MNIST: {acc:.2f}%")
