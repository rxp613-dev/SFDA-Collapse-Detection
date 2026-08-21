# SFDA-Collapse-Detection

Source-Free Domain Adaptation (SFDA) for Bearing Fault Diagnosis with Model Collapse Detection

## 📋 项目概述

本项目实现了基于源无域适应（SFDA）的轴承故障诊断方法，并提出了基于类别偏移的崩溃检测框架。通过对多种SFDA方法在工业噪声条件下的系统性审计，建立了学习率敏感性的两级分类体系。

## 🔬 主要贡献

1. **系统性实证审计**：在CWRU数据集上对4种经典SFDA方法（SHOT、TENT、NRC、SAR）和2种2024年方法进行了约960次实验
2. **两级学习率敏感性分类**：
   - LR-Robust方法：TENT、SAR（在噪声下保持稳定）
   - LR-Sensitive方法：SHOT、NRC（需要谨慎调参）
3. **崩溃检测框架**：提出基于类别偏移的检测器（AUC=0.809）
4. **工业部署指南**：提供实用的部署建议和回退机制

## 📊 数据集

- **CWRU Bearing Dataset**：凯斯西储大学轴承数据中心
- **PU Dataset**：帕德博恩大学轴承数据集（用于跨数据集验证）

## 🚀 快速开始

### 环境要求

```bash
pip install -r requirements.txt
```

### 训练源模型

```bash
python scripts/train_source.py --dataset cwru --save_dir checkpoints/
```

### 运行SFDA适应

```bash
# SHOT方法
python scripts/run_sfda.py --method shot --source_checkpoint checkpoints/source.pth --target_domain 3HP

# TENT方法
python scripts/run_sfda.py --method tent --source_checkpoint checkpoints/source.pth --target_domain 3HP
```

### 崩溃检测

```bash
python scripts/collapse_detection.py --results_dir results/ --output figures/
```

## 📁 项目结构

```
├── src/                    # 核心源代码
│   ├── methods/           # SFDA方法实现（SHOT, TENT, NRC, SAR）
│   ├── models/            # 模型架构
│   ├── monitoring/        # 崩溃检测模块
│   ├── utils/             # 工具函数
│   └── data/              # 数据处理
├── scripts/               # 实验脚本
├── configs/               # 配置文件
└──  data/                  # 数据目录
```

## 📈 主要结果

### 学习率敏感性分类

| 方法 | 类型 | 默认LR准确率 | 最优LR准确率 |
|------|------|-------------|-------------|
| TENT | LR-Robust | 87.61% ± 0.35% | 89.85% |
| SAR | LR-Robust | 90.80% ± 0.10% | 91.35% |
| SHOT | LR-Sensitive | 91.06% ± 8.39% | 93.95% ± 0.21% |
| NRC | LR-Sensitive | 77.09% ± 6.90% | 84.76% |

### 崩溃检测性能

- **AUC**: 0.809（L1-only）
- **最佳阈值**: 0.930（Youden-optimal）
- **计算开销**: 仅4% GPU开销

## 📖 论文

本文已投稿至Machines (MDPI)期刊。

**标题**: An Empirical Audit and Class-Shift-Based Collapse Detection Framework for Source-Free Domain Adaptation in Bearing Fault Diagnosis

**作者**: Chaoya Song, Xiaoping Ren

## 📄 许可证

MIT License

## 📧 联系方式

如有问题，请联系：
- Xiaoping Ren (rxp613@gmail.com)
