"""Run streamlined experiments: SegFormer-B0 + Boundary Enhancement + Ablation.

Plan:
  1. SegFormer-B0 (reproduction): 50 epochs
  2. SegFormer-B0 + Boundary Head lambda=0.4 (our improvement): 50 epochs
  3. Ablation lambda=0, 0.2, 0.6: 30 epochs each

UNet/DeepLabV3+ results taken from literature for comparison.
"""
import os
import sys
import subprocess
import torch

sys.path.insert(0, os.path.dirname(__file__))

from experiments.config import OUTPUT_DIR, DEVICE
from experiments.dataset import download_camvid
from experiments.utils import set_seed


def run_experiment(model_arg, boundary_lambda=None, epochs=50):
    """Run a single experiment via main.py."""
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "main.py"),
           "--model", model_arg, "--epochs", str(epochs)]
    if boundary_lambda is not None:
        cmd.extend(["--lambda", str(boundary_lambda)])
    print(f"\n{'#'*60}")
    print(f"# Running: {' '.join(cmd)}")
    print(f"{'#'*60}\n")
    subprocess.run(cmd, check=True)


def print_summary():
    """Print summary of collected results."""
    results_dir = os.path.join(OUTPUT_DIR, "results")
    if not os.path.exists(results_dir):
        print("No results yet.")
        return

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'mIoU%':>8} {'PA%':>8} {'BF1%':>8} {'Params':>8} {'FPS':>7}")
    print("-" * 70)

    order = [
        ("segformer", "SegFormer-B0"),
        ("segformer_boundary_lambda0.0", "+Boundary lambda=0"),
        ("segformer_boundary_lambda0.2", "+Boundary lambda=0.2"),
        ("segformer_boundary_lambda0.4", "+Boundary lambda=0.4"),
        ("segformer_boundary_lambda0.6", "+Boundary lambda=0.6"),
    ]

    for fname, label in order:
        path = os.path.join(results_dir, f"{fname}_test.pt")
        if os.path.exists(path):
            r = torch.load(path, map_location="cpu", weights_only=False)
            print(f"{label:<35} {r.get('mIoU',0):>7.2f} {r.get('PA',0):>7.2f} "
                  f"{r.get('Boundary_F1',0):>7.2f} {r.get('Params_M',0):>7.1f} {r.get('FPS',0):>6.1f}")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--segformer-epochs", type=int, default=50)
    parser.add_argument("--ablation-epochs", type=int, default=30)
    args = parser.parse_args()

    set_seed(42)
    print(f"Device: {DEVICE}")

    # Ensure dataset exists
    download_camvid()

    # ================================================================
    # Phase 1: SegFormer-B0 reproduction (50 epochs)
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: SegFormer-B0 Reproduction (50 epochs)")
    print("=" * 60)
    run_experiment("segformer", epochs=args.segformer_epochs)

    # ================================================================
    # Phase 2: Our model - SegFormer+Boundary lambda=0.4 (50 epochs)
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: SegFormer + Boundary Head lambda=0.4 (50 epochs)")
    print("=" * 60)
    run_experiment("segformer_boundary", boundary_lambda=0.4, epochs=args.segformer_epochs)

    # ================================================================
    # Phase 3: Ablation - lambda = 0, 0.2, 0.6 (30 epochs each)
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Ablation Studies (30 epochs each)")
    print("=" * 60)
    for lam in [0.0, 0.2, 0.6]:
        run_experiment("segformer_boundary", boundary_lambda=lam, epochs=args.ablation_epochs)

    # ================================================================
    # Summary
    # ================================================================
    print_summary()


if __name__ == "__main__":
    main()
