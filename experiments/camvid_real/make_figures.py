import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("experiments/camvid_real/outputs")
FIG = Path("experiments/camvid_real/figures")
FIG.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


summary = read_csv(ROOT / "summary.csv")
methods = [r["method"] for r in summary]
miou = [float(r["mIoU"]) * 100 for r in summary]
bf1 = [float(r["BoundaryF1"]) * 100 for r in summary]
pa = [float(r["PA"]) * 100 for r in summary]

plt.figure(figsize=(7, 4))
x = range(len(methods))
plt.bar(x, miou, label="mIoU")
plt.bar([i + 0.28 for i in x], bf1, width=0.28, label="Boundary F1")
plt.xticks([i + 0.14 for i in x], methods, rotation=15, ha="right")
plt.ylabel("Score (%)")
plt.title("CamVid Tiny Test Metrics")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "camvid_tiny_metrics.png", dpi=200)
plt.close()

plt.figure(figsize=(7, 4))
for method in methods:
    hist = read_csv(ROOT / method / "history.csv")
    epochs = [int(r["epoch"]) for r in hist]
    vals = [float(r["mIoU"]) * 100 for r in hist]
    plt.plot(epochs, vals, marker="o", markersize=3, label=method)
plt.xlabel("Epoch")
plt.ylabel("mIoU (%)")
plt.title("CamVid Tiny Training Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "camvid_tiny_training_curves.png", dpi=200)
plt.close()

with (FIG / "real_experiment_table.tex").open("w", encoding="utf-8") as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\caption{CamVid Tiny真实测试集实验结果}\n")
    f.write("\\label{tab:camvid_tiny_real}\n")
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

print("Wrote figures to", FIG)
