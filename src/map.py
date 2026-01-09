import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 整理实验数据 (基于你提供的 Log)
models = ['Baseline', 'Res_Skip', 'Dense_Fusion', 'MultiScale', 'SPP_Multi', 'STN_Spatial', 'GAP_Light']
accs = [91.76, 92.20, 91.82, 91.10, 90.71, 90.52, 86.89]  # 准确率
params = [421642, 513994, 389678, 424754, 192266, 55706, 93962]  # 参数量

# 设置绘图风格 (如果报错提示没有这个风格，可以改成 'ggplot' 或直接注释掉)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot') # 备用风格

plt.figure(figsize=(14, 6))

# === 图1: 参数量 vs 准确率 散点图 (Efficiency Map) ===
plt.subplot(1, 2, 1)
colors = ['grey', 'red', 'green', 'blue', 'orange', 'purple', 'brown']

for i, model in enumerate(models):
    plt.scatter(params[i], accs[i], s=150, c=colors[i], label=model, alpha=0.8, edgecolors='w')
    # 给每个点加标签，位置稍作调整避免重叠
    plt.text(params[i], accs[i]+0.2, model, fontsize=9, ha='center', fontweight='bold')

plt.xlabel('Number of Parameters (Model Size)', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.title('Model Efficiency: Accuracy vs. Size', fontsize=14, fontweight='bold')
# 调整X轴刻度显示，使其更易读（例如用 k 表示千）
current_values = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:.0f}k'.format(x/1000) for x in current_values])
plt.grid(True, linestyle='--', alpha=0.7)

# === 图2: 准确率柱状图 ===
plt.subplot(1, 2, 2)
bars = plt.bar(models, accs, color=colors, alpha=0.7)
# 动态设置Y轴范围，让差异看起来更明显
plt.ylim(min(accs)-1, max(accs)+1)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)

# 在柱子上标数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

# 自动调整布局防止重叠
plt.tight_layout()

# =========== 核心修改部分 ===========
# 定义保存路径
save_filename = 'model_comparison_results.png'
# 获取当前脚本所在的绝对路径
save_path = os.path.join(os.getcwd(), save_filename)

# 保存图片
# dpi=300 表示高分辨率（适合放在论文/报告中）
# bbox_inches='tight' 可以去除图片周围多余的白边
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"--------------------------------------------------")
print(f"图表已成功保存为图片文件！")
print(f"保存路径: {save_path}")
print(f"请在左侧文件浏览器中刷新并查看该图片。")
print(f"--------------------------------------------------")

# 注释掉 plt.show()，这样就不会在服务器端尝试弹窗显示
# plt.show()