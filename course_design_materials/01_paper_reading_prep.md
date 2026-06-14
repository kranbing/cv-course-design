# 论文精读准备

## 选题

基于 Transformer 的城市道路场景语义分割复现与边界增强改进。

研究目标是比较近五年语义分割模型从 CNN、多尺度 Transformer、统一 mask 分类到视觉基础模型的发展路线，并重点围绕 SegFormer 进行可解释复现和边界增强改进。

## 精读论文清单

| 编号 | 论文 | 年份 | 方向定位 | 官方代码 | 精读重点 |
|---|---|---:|---|---|---|
| P1 | SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers | 2021 | 高效 Transformer 语义分割 | https://github.com/NVlabs/SegFormer | MiT 编码器、无位置编码、多尺度输出、MLP 解码器 |
| P2 | Masked-attention Mask Transformer for Universal Image Segmentation | 2022 | 统一语义/实例/全景分割 | https://github.com/facebookresearch/Mask2Former | mask classification、masked attention、query 与 mask 的关系 |
| P3 | PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers | 2023 | 实时道路场景分割 | https://github.com/XuJiacong/PIDNet | P/I/D 三分支、边界分支、边界注意力、速度精度权衡 |
| P4 | Segment Anything | 2023 | promptable segmentation 基础模型 | https://github.com/facebookresearch/segment-anything | prompt encoder、image encoder、mask decoder、零样本泛化 |

## 每篇论文的精读问题

### P1 SegFormer

1. 为什么传统 ViT 的固定位置编码不适合密集预测？
2. MiT 编码器如何生成 1/4、1/8、1/16、1/32 四级特征？
3. Efficient Self-Attention 中 `sr_ratio` 如何降低注意力复杂度？
4. MLP 解码器为什么比复杂 CNN decoder 更轻量？
5. SegFormer-B0 到 B5 的区别体现在哪些超参数上？

报告可写结论：SegFormer 的核心贡献不是单纯使用 Transformer，而是把层次化特征、空间降采样注意力和轻量 MLP 解码组合起来，在精度和效率之间取得较好平衡。

### P2 Mask2Former

1. 语义分割为什么可以从逐像素分类转化为 mask 分类？
2. masked attention 与普通 cross-attention 的区别是什么？
3. 为什么同一架构可以支持 semantic、instance、panoptic 三类任务？
4. 其训练成本和模型复杂度为何高于 SegFormer？

报告可写结论：Mask2Former 代表了分割任务从 per-pixel classification 向 mask classification 的范式转移，但课程设计中完整复现成本较高，因此作为理论对比模型。

### P3 PIDNet

1. 论文如何把实时语义分割中的两分支网络类比为 PI 控制器？
2. overshoot 问题在分割结果中体现为什么现象？
3. D 分支为什么适合建模边界？
4. boundary attention 如何辅助融合细节和语义？
5. PIDNet-S、M、L 的速度和精度差异如何解释？

报告可写结论：PIDNet 对本文改进有直接启发。本文不是复现 PIDNet，而是借鉴其边界辅助思想，在 SegFormer 解码器上增加轻量边界监督。

### P4 Segment Anything

1. SAM 的 promptable segmentation 与普通语义分割有何区别？
2. prompt encoder、image encoder、mask decoder 分别承担什么功能？
3. SAM 为什么不直接输出 Cityscapes 类别语义？
4. SAM 在本文中适合作为哪种扩展讨论？

报告可写结论：SAM 体现了视觉基础模型趋势，但它解决的是提示驱动的通用掩码生成，不直接替代有类别标签的道路场景语义分割。

## 精读输出物

每篇论文后续报告中建议形成 4 个固定段落：

1. 研究问题与动机。
2. 网络结构与关键公式。
3. 算法流程与代码实现对应关系。
4. 优点、局限及其对本文实验的启发。

