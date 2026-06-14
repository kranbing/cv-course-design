# CamVid Benchmark 方法对比实验

- 数据集：data\camvid
- 划分：fastai CamVid valid.txt
- 训练集：600 张
- 测试/验证集：101 张
- 输入尺寸：96 x 128
- 训练轮数：20
- batch size：8

| 方法 | Best Epoch | mIoU | PA | Boundary F1 | FPS |
|---|---:|---:|---:|---:|---:|
| UNetSmall | 20 | 27.41% | 86.34% | 54.98% | 311.6 |
| TinySegFormer | 20 | 25.05% | 83.00% | 46.30% | 189.5 |
| TinySegFormer_Boundary | 19 | 26.30% | 85.34% | 46.81% | 204.9 |
