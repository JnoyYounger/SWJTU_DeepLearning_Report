# SWJTU 深度学习课程设计报告

> 基于 PyTorch 的 Fashion-MNIST 图像分类模型优化实验

本项目是西南交通大学（SWJTU）深度学习课程的课程设计实现，旨在通过修改神经网络结构，系统探究不同优化技术对 **Fashion-MNIST** 图像分类任务性能的影响。

---

## 📌 项目简介

- **数据集**：[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)（10 类服装图像，28×28 灰度图）
- **基准模型**：`Net_Baseline`（简单卷积神经网络）
- **改进方向**：
  - 批归一化（Batch Normalization）
  - Dropout 正则化
  - 残差连接（Residual Block）
  - 深度可分离卷积（Depthwise Separable Convolution）
  - 注意力机制（如 SE Block）
- **实验目标**：对比不同结构对准确率、收敛速度与泛化能力的影响
- **框架**：PyTorch
- **可复现性**：固定随机种子为 `42`

---

## 📂 项目结构
