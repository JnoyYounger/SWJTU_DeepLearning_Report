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
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        # 分支1: 1x1 卷积 (捕捉跨通道特征)
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 分支2: 3x3 卷积 (捕捉空间特征)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 两个分支输出通道数各为 out_channels，拼接后总数为 out_channels * 2

    def forward(self, x):
        out1 = F.relu(self.branch1(x))
        out2 = F.relu(self.branch2(x))
        return torch.cat([out1, out2], dim=1) # 在通道维度拼接

class Net_MultiScale(nn.Module):
    def __init__(self):
        super(Net_MultiScale, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 输入20，每个分支输出16，拼接后是32
        self.inception = InceptionBlock(in_channels=20, out_channels=16) 
        
        self.conv_final = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 14x14
        x = self.inception(x)                # 14x14, channel变为32
        x = self.pool(x)                     # 7x7
        x = F.relu(self.conv_final(x))       # 7x7, channel变为64
        
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
    print("模型结构: Net_MultiScale")
    
    # 实例化模型
    model = Net_MultiScale().to(DEVICE)
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
    evaluate_detailed(model, DEVICE, test_loader, model_name="Net_MultiScale")