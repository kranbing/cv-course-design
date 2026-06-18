# CamVid Tiny 真实实验结果

## 实验说明

本实验使用 fastai 提供的 `camvid_tiny` 数据集子集，包含 100 张道路场景图像及像素级标签。实验采用固定随机种子 42，将数据按 80/20 划分为训练集和测试集，即 80 张训练图像、20 张测试图像。输入尺寸为 `96 x 128`，类别数为 32，其中 `Void` 类在 mIoU 和 PA 计算中忽略。

由于完整 CamVid 数据包下载过程中出现截断，本实验不是正式 CamVid benchmark，而是 CamVid 子集上的真实训练与测试。它可以用于替换报告中“完全模拟”的部分，但需要在报告中明确写作“CamVid Tiny 子集实验”。

## 对比方法

| 方法 | 含义 |
|---|---|
| UNetSmall | 小型 CNN 编码器-解码器基线 |
| TinySegFormer | 采用重叠 patch embedding、多头自注意力和 MLP 解码思想的轻量 SegFormer 风格复现 |
| TinySegFormer_Boundary | 在 TinySegFormer 上加入边界预测头和边界 BCE 损失，作为本文改进方法 |

## 测试结果

| 方法 | Best Epoch | mIoU | PA | Boundary F1 | FPS |
|---|---:|---:|---:|---:|---:|
| UNetSmall | 29 | 16.55% | 77.72% | 36.54% | 927.5 |
| TinySegFormer | 29 | 17.08% | 75.63% | 35.01% | 484.4 |
| TinySegFormer_Boundary | 29 | 17.17% | 74.73% | 38.00% | 529.5 |

## 结果分析

1. TinySegFormer 的 mIoU 略高于 UNetSmall，说明即使在小数据子集上，引入全局注意力和多尺度融合也能带来一定收益。
2. TinySegFormer_Boundary 的 mIoU 进一步从 17.08% 提升到 17.17%，提升幅度较小，但 Boundary F1 从 35.01% 提升到 38.00%，说明边界监督主要改善边缘区域预测。
3. UNetSmall 的 PA 较高，但 mIoU 较低，说明其更容易预测大面积高频类别，而对小类别和细粒度类别的均衡识别不足。
4. CamVid Tiny 数据量很小，类别数却有 32 类，训练集仅 80 张，因此 mIoU 绝对值偏低是合理现象。该实验更适合验证方法趋势，而不是作为正式 benchmark 性能声明。

## 生成文件

- 训练脚本：`experiments/camvid_real/train_camvid_tiny.py`
- 结果表：`experiments/camvid_real/outputs/summary.csv`
- 训练日志：`experiments/camvid_real/outputs/*/history.csv`
- 指标柱状图：`experiments/camvid_real/figures/camvid_tiny_metrics.png`
- 训练曲线图：`experiments/camvid_real/figures/camvid_tiny_training_curves.png`
- LaTeX 表格片段：`experiments/camvid_real/figures/camvid_tiny_experiment_table.tex`
