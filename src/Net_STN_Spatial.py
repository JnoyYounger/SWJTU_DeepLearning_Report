import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
# ==========================================
# 1. 实验设置 (固定随机种子以保证结果可靠)
# ==========================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)  # 使用 42 作为种子

# 超参数配置
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5      # 课程设计演示用5轮即可，追求高精度可改为10-20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据集准备 (Fashion-MNIST)
# ==========================================
# 转换操作：转为Tensor并标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 归一化有助于模型更快收敛
])

print("正在下载/加载数据集...")
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ==========================================
# 3. 定义基准网络结构 (Net_Baseline)
# ==========================================
class Net_STN_Spatial(nn.Module):
    def __init__(self):
        super(Net_STN_Spatial, self).__init__()
        
        # === 1. 定位网络 (Localization Network) ===
        # 负责观察图片，计算出需要旋转/平移的参数
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # 回归出仿射变换的 2x3 矩阵参数
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # 初始化权重：保证初始状态下不做变换（Identity Transform）
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # === 2. 常规分类网络 ===
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 10)

    # STN 核心逻辑
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # 生成网格并采样
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_transformed = F.grid_sample(x, grid, align_corners=True)
        return x_transformed

    def forward(self, x):
        # 第一步：矫正图片
        x = self.stn(x) 
        
        # 第二步：常规分类
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        return x

# ==========================================
# 4. 训练与测试函数
# ==========================================
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 300 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] \tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n')
    return acc

def evaluate_detailed(model, device, test_loader, model_name="Default"):
    print(f"\n正在生成 {model_name} 的详细评估报告...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # --- 修改点 1: 修正语法错误 ---
            # 先取argmax拿到索引，再转到cpu，最后转numpy
            pred = output.argmax(dim=1).cpu().numpy() 
            
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred)
            
    # 1. 打印详细报告 (包含 Precision, Recall, F1-score)
    print(f"\nClassification Report ({model_name}):")
    # digits=4 让小数显示更精确
    print(classification_report(all_targets, all_preds, digits=4))
    
    # 2. 画混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # --- 修改点 2: 保存图片而不是显示 ---
    save_filename = f'confusion_matrix_{model_name}.png'
    save_path = os.path.join(os.getcwd(), save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 非常重要：关闭画布，防止和之前的图重叠
    plt.close()
    
    print(f"混淆矩阵已保存为: {save_filename}")


# ==========================================
# 5. 主程序执行
# ==========================================
if __name__ == '__main__':
    print(f"使用设备: {DEVICE}")
    print("模型结构: Net_STN_Spatial")
    
    # 实例化模型
    model = Net_STN_Spatial().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 记录参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    start_time = time.time()
    
    # 开始训练循环
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)
        
    end_time = time.time()
    print(f"训练完成! 总耗时: {end_time - start_time:.2f} 秒")
    evaluate_detailed(model, DEVICE, test_loader, model_name="Net_STN_Spatial")