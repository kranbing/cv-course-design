import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("experiments/camvid_benchmark/outputs")
FIG = Path("experiments/camvid_benchmark/figures")
FIG.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


summary = read_csv(ROOT / "summary.csv")
metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))

methods = [r["method"] for r in summary]
miou = [float(r["mIoU"]) * 100 for r in summary]
pa = [float(r["PA"]) * 100 for r in summary]
bf1 = [float(r["BoundaryF1"]) * 100 for r in summary]

plt.figure(figsize=(7.5, 4.4))
x = list(range(len(methods)))
width = 0.25
plt.bar([i - width for i in x], miou, width=width, label="mIoU")
plt.bar(x, pa, width=width, label="PA")
plt.bar([i + width for i in x], bf1, width=width, label="Boundary F1")
plt.xticks(x, methods, rotation=12, ha="right")
plt.ylabel("Score (%)")
plt.title("CamVid Benchmark Method Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "camvid_benchmark_metrics.png", dpi=200)
plt.close()

plt.figure(figsize=(7.5, 4.4))
for method in methods:
    hist = read_csv(ROOT / method / "history.csv")
    epochs = [int(r["epoch"]) for r in hist]
    vals = [float(r["mIoU"]) * 100 for r in hist]
    plt.plot(epochs, vals, marker="o", markersize=3, label=method)
plt.xlabel("Epoch")
plt.ylabel("mIoU (%)")
plt.title("CamVid Benchmark mIoU Curves")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "camvid_benchmark_miou_curves.png", dpi=200)
plt.close()

with (FIG / "benchmark_table.tex").open("w", encoding="utf-8") as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\caption{完整CamVid划分上的方法对比实验结果}\n")
    f.write("\\label{tab:camvid_benchmark_real}\n")
    f.write("\\begin{tabular}{lccccc}\n")
    f.write("\\toprule\n")
    f.write("方法 & Best Epoch & mIoU(\\%) & PA(\\%) & Boundary F1(\\%) & FPS \\\\\n")
    f.write("\\midrule\n")
    for r in summary:
        f.write(
            f"{r['method']} & {r['best_epoch']} & "
            f"{float(r['mIoU']) * 100:.2f} & {float(r['PA']) * 100:.2f} & "
            f"{float(r['BoundaryF1']) * 100:.2f} & {float(r['FPS']):.1f} \\\\\n"
        )
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

with (FIG / "benchmark_summary.md").open("w", encoding="utf-8") as f:
    f.write("# CamVid Benchmark 方法对比实验\n\n")
    f.write(f"- 数据集：{metadata['dataset']}\n")
    f.write(f"- 划分：{metadata['split']}\n")
    f.write(f"- 训练集：{metadata['train_count']} 张\n")
    f.write(f"- 测试/验证集：{metadata['test_count']} 张\n")
    f.write(f"- 输入尺寸：{metadata['image_size'][0]} x {metadata['image_size'][1]}\n")
    f.write(f"- 训练轮数：{metadata['epochs']}\n")
    f.write(f"- batch size：{metadata['batch_size']}\n\n")
    f.write("| 方法 | Best Epoch | mIoU | PA | Boundary F1 | FPS |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for r in summary:
        f.write(
            f"| {r['method']} | {r['best_epoch']} | "
            f"{float(r['mIoU']) * 100:.2f}% | {float(r['PA']) * 100:.2f}% | "
            f"{float(r['BoundaryF1']) * 100:.2f}% | {float(r['FPS']):.1f} |\n"
        )

print("Wrote benchmark figures to", FIG)
