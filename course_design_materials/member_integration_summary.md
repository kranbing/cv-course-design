# 成员实验材料归并与可用性评估

## 已归并内容

- 成员实验代码已整理到 `experiments/member_segformer_ablation/`。
- 成员实验说明已整理为 `experiments/member_segformer_ablation/README_实验说明.md`。
- 成员实验结果表已整理为 `course_design_materials/member_ablation_results.csv`。
- 成员 LaTeX 表格/公式片段已整理为 `course_design_materials/member_report_insert_snippets.tex`。
- 成员报告草稿已保留为 `course_design_materials/member_main_report_draft.tex`，未覆盖当前主报告。
- 成员输出图表已整理到 `experiments/member_segformer_ablation/outputs/figures/`。
- 成员训练曲线日志 `history.json` 已整理到 `experiments/member_segformer_ablation/outputs/logs/`。
- 成员 CamVid 目录结构数据已整理到 `data/CamVid_member_segformer/`，该目录不纳入 Git。
- 成员 checkpoint 与 `.pt` 测试结果已保留在成员实验目录的 `outputs/` 下，但按 `.gitignore` 不纳入 Git。

## 成员实验新增信息

- 模型代码包含 UNet、DeepLabV3+、SegFormer-B0、SegFormer-B0 + Boundary Head。
- SegFormer-B0 在 CamVid 上完成 50 epoch 真实训练，记录 mIoU 约 51.0%。
- 改进模型 SegFormer-B0 + Boundary Head 完成边界增强实验，最佳设置为 `lambda=0.4`。
- 消融实验包含 `lambda=0.0`、`0.2`、`0.4`、`0.6` 四组。
- 已提供主对比表、消融表、训练曲线图、柱状对比图和分割可视化图。

## 可用性评估

- 数据路径已适配为优先读取 `data/CamVid_member_segformer/`，若不存在再回退到成员子项目内部的 `data/CamVid/`。
- 成员子项目入口为 `experiments/member_segformer_ablation/main.py`。
- 当前环境测试中，配置导入成功，数据路径存在，输出路径存在。
- 当前环境缺少 `albumentations`，因此直接运行 `main.py --help` 会在导入数据集模块时报错。
- 已补充 `experiments/member_segformer_ablation/requirements.txt`，安装依赖后可继续运行训练/评估脚本。

## 后续报告整合建议

- 将 `course_design_materials/member_report_insert_snippets.tex` 中的主对比表和消融表合入 `template/main_report.tex`。
- 将 `experiments/member_segformer_ablation/outputs/figures/` 中的曲线图、柱状图和分割对比图插入实验章节。
- 在报告中明确区分：文献参考结果、成员真实训练结果、当前项目已有轻量 benchmark 结果。
- 若最终报告采用成员的 SegFormer-B0 50 epoch 结果，应同步修改实验设置、数据集划分和指标说明，避免与当前 `96 x 128` 轻量 benchmark 混写。
