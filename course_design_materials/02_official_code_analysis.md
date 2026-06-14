# 官方代码分析

## SegFormer 官方代码

仓库：https://github.com/NVlabs/SegFormer

论文原文和仓库均给出官方 PyTorch 实现。仓库基于 MMSegmentation v0.13.0，核心目录包括：

- `local_configs/segformer/`：不同模型尺寸和数据集的配置文件。
- `mmseg/models/backbones/mix_transformer.py`：MiT 编码器。
- `mmseg/models/decode_heads/segformer_head.py`：SegFormer MLP 解码头。
- `tools/train.py`：训练入口。
- `tools/test.py`：测试入口。
- `demo/image_demo.py`：单图推理可视化入口。

### 编码器实现对应关系

`mix_transformer.py` 中包含以下关键模块：

- `OverlapPatchEmbed`：用卷积完成重叠 patch embedding。第一层通常使用 kernel=7、stride=4，后续层使用 kernel=3、stride=2。
- `Attention`：实现带空间缩减的 self-attention。当 `sr_ratio > 1` 时，先对 key/value 特征进行卷积降采样，再计算注意力。
- `Mlp`：线性层后接 depth-wise convolution，引入局部空间信息。
- `MixVisionTransformer`：串联四个 stage，输出四级特征。
- `mit_b0` 到 `mit_b5`：通过 `embed_dims`、`depths`、`num_heads` 等参数定义不同规模模型。

### 解码器实现对应关系

`segformer_head.py` 中的 `SegFormerHead` 处理四级特征 `c1,c2,c3,c4`：

1. 每一级特征先经过线性 MLP 投影到统一维度。
2. `c2,c3,c4` 双线性上采样到 `c1` 的空间尺寸。
3. 四级特征在通道维拼接。
4. 使用 `1x1 Conv` 融合。
5. 使用 `1x1 Conv` 输出类别 logits。

该结构适合作为本文主要复现对象，因为实现逻辑清晰，训练流程标准，改进边界分支也比较自然。

## Mask2Former 官方代码

仓库：https://github.com/facebookresearch/Mask2Former

仓库已于 2025-01-01 归档为只读，但仍可用于课程报告分析。代码基于 Detectron2，核心目录包括：

- `configs/`：COCO、ADE20K、Cityscapes 等配置。
- `mask2former/`：模型主体，包括 transformer decoder、criterion、matcher、pixel decoder 等。
- `datasets/`：数据集注册和预处理。
- `train_net.py`：训练入口。
- `demo/`：可视化推理。

Mask2Former 的代码复杂度明显高于 SegFormer，涉及 query-based decoder、Hungarian matching、多任务损失等。本文适合将其作为理论分析和范式对比，不作为主要复现实验对象。

## PIDNet 官方代码

仓库：https://github.com/XuJiacong/PIDNet

仓库给出 Cityscapes 和 CamVid 上的训练、测试脚本以及预训练权重链接。核心目录包括：

- `models/`：PIDNet 主干模型。
- `configs/`：Cityscapes、CamVid 不同模型规模配置。
- `datasets/`：数据读取与标签处理。
- `tools/train.py`：训练入口。
- `tools/eval.py`：评估入口。
- `tools/test.py`：测试与可视化入口。

论文和 README 给出的关键信息：

- PIDNet-S 在 Cityscapes test 上约 78.6% mIoU、93.2 FPS。
- PIDNet-M 在 Cityscapes test 上约 79.8% mIoU、42.2 FPS。
- PIDNet-L 在 Cityscapes test 上约 80.6% mIoU、31.1 FPS。
- 在 CamVid 上，PIDNet-S test 约 80.1% mIoU、153.7 FPS。

本文改进方案直接借鉴 PIDNet 的边界分支思想，但实现上不复制 PIDNet 三分支结构，而是在 SegFormer 的 decoder 后增加轻量边界 head。

## Segment Anything 官方代码

仓库：https://github.com/facebookresearch/segment-anything

仓库提供推理代码、checkpoint 下载和 notebook 示例，核心目录包括：

- `segment_anything/`：SAM 模型主体。
- `notebooks/`：prompt 推理和自动掩码生成示例。
- `scripts/amg.py`：自动掩码生成命令行入口。
- `demo/`：ONNX 和 Web demo。

SAM 的最小调用流程为：

1. 通过 `sam_model_registry` 加载模型。
2. 用 `SamPredictor.set_image()` 编码图像。
3. 用点、框或 mask prompt 调用 `predict()` 得到候选 mask。
4. 或用 `SamAutomaticMaskGenerator.generate()` 生成整图候选 mask。

SAM 不直接输出道路、车辆、行人等语义类别，因此本文将它作为近年分割基础模型的扩展讨论，而不是主要对比模型。

## 本文模拟复现的代码映射

报告中可将复现代码描述为以下抽象模块：

```text
Input image
  -> data augmentation
  -> SegFormer MiT encoder
  -> MLP decoder
  -> semantic logits
  -> boundary enhancement head
  -> semantic loss + boundary loss
  -> mIoU / PA / FPS analysis
```

其中已有工作包括：SegFormer 的 MiT encoder 与 MLP decoder，PIDNet 的边界辅助思想。本文自拟工作包括：将边界监督接入 SegFormer decoder、设计联合损失权重并进行模拟消融分析。

