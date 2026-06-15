"""Real-time training progress monitor."""
import os
import time
import sys
import io

# Fix encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUTPUT_FILE = r"C:\Users\NEHC\AppData\Local\Temp\claude\D--Downloads-----\5d9444c9-3bd9-45ad-a0d2-a1ec67240408\tasks\bjlx9nfn7.output"
TOTAL_EPOCHS = 50  # segformer + boundary epochs; ablation runs 30

# Total experiments in order
ALL_MODELS = [
    ("unet", "UNet baseline"),
    ("deeplabv3", "DeepLabV3+ baseline"),
    ("segformer", "SegFormer-B0 reproduction"),
    ("segformer_boundary_lambda0.0", "SegFormer+Boundary λ=0"),
    ("segformer_boundary_lambda0.2", "SegFormer+Boundary λ=0.2"),
    ("segformer_boundary_lambda0.4", "SegFormer+Boundary λ=0.4"),
    ("segformer_boundary_lambda0.6", "SegFormer+Boundary λ=0.6"),
]


def parse_log():
    """Parse training log and return current state."""
    if not os.path.exists(OUTPUT_FILE):
        return None

    with open(OUTPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    lines = content.split("\n")

    # Find current model
    current_model = "unknown"
    current_idx = -1
    for i, (key, name) in enumerate(ALL_MODELS):
        if f"Training {key}" in content:
            current_model = name
            current_idx = i

    # Extract all epoch results for current model
    epoch_lines = []
    in_current = True
    for line in reversed(lines):
        if line.startswith("Epoch"):
            epoch_lines.append(line)

    # Get latest epoch
    latest = {}
    if epoch_lines:
        parts = epoch_lines[0].split("|")
        for p in parts:
            p = p.strip()
            if p.startswith("Epoch"):
                latest["epoch"] = int(p.split()[1])
            elif p.startswith("Train Loss:"):
                latest["train_loss"] = float(p.split(":")[1].strip())
            elif p.startswith("Val Loss:"):
                latest["val_loss"] = float(p.split(":")[1].strip())
            elif p.startswith("Val mIoU:"):
                latest["miou"] = float(p.split(":")[1].strip())
            elif p.startswith("Val PA:"):
                latest["pa"] = float(p.split(":")[1].strip())
            elif p.startswith("Val BF1:"):
                latest["bf1"] = float(p.split(":")[1].strip())

    # Find best mIoU so far
    best_miou = 0
    best_epoch = 0
    for line in epoch_lines:
        if "Best mIoU:" in line:
            try:
                best_miou = float(line.split("Best mIoU:")[1].split()[0])
                best_epoch = int(line.split("epoch")[1].strip())
            except (ValueError, IndexError):
                pass
    if best_miou == 0 and epoch_lines:
        for line in epoch_lines:
            if "Val mIoU:" in line:
                try:
                    miou_str = line.split("Val mIoU:")[1].split("|")[0].strip()
                    miou = float(miou_str)
                    if miou > best_miou:
                        best_miou = miou
                except (ValueError, IndexError):
                    pass

    # Completed model checkpoints
    completed = []
    for key, name in ALL_MODELS:
        if f"Best mIoU" in content and key in content:
            # Check if next model started
            pass

    return {
        "current_model": current_model,
        "current_idx": current_idx,
        "total_models": len(ALL_MODELS),
        "latest": latest,
        "best_miou": best_miou,
        "epoch_lines": epoch_lines,
    }


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def progress_bar(value, max_val, width=30):
    filled = int(width * value / max_val) if max_val > 0 else 0
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {value}/{max_val}"


def main():
    try:
        interval = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    except ValueError:
        interval = 10

    print(f"Monitoring training progress (refresh every {interval}s). Ctrl+C to exit.\n")

    last_len = 0
    try:
        while True:
            info = parse_log()
            if info is None:
                print("Waiting for output file...")
                time.sleep(interval)
                continue

            latest = info["latest"]
            epoch = latest.get("epoch", 0)

            # Build display
            display = []
            display.append("=" * 60)
            display.append("  TRAINING MONITOR  (GPU: RTX 4060 8GB)")
            display.append("=" * 60)
            display.append(f"  Model: {info['current_model']} ({info['current_idx']+1}/{info['total_models']})")
            display.append(f"  Epoch: {progress_bar(epoch, TOTAL_EPOCHS)}")
            display.append("-" * 60)

            if latest:
                tl = latest.get("train_loss", 0)
                vl = latest.get("val_loss", 0)
                miou = latest.get("miou", 0) * 100
                pa = latest.get("pa", 0) * 100
                bf1 = latest.get("bf1", 0) * 100

                display.append(f"  Train Loss : {tl:.4f}")
                display.append(f"  Val Loss   : {vl:.4f}")
                display.append(f"  Val mIoU   : {miou:.2f}%")
                display.append(f"  Val PA     : {pa:.2f}%")
                if bf1 > 0:
                    display.append(f"  Val BF1    : {bf1:.2f}%")

            display.append("-" * 60)

            # Show mIoU trend (last 10 epochs)
            trend_lines = info["epoch_lines"][:min(10, len(info["epoch_lines"]))]
            if trend_lines:
                display.append("  Recent mIoU trend:")
                for line in reversed(trend_lines[:8]):
                    if "Val mIoU:" in line:
                        try:
                            ep = line.split("Epoch")[1].split("|")[0].strip()
                            miou = line.split("Val mIoU:")[1].split("|")[0].strip()
                            display.append(f"    Epoch {ep:>4s} | mIoU: {float(miou)*100:5.1f}%")
                        except (ValueError, IndexError):
                            pass
                display.append("-" * 60)

            # ETA estimate
            SEC_PER_EPOCH = 90  # ~90 seconds per epoch with batch_size=12
            if epoch > 0:
                remaining_epochs = TOTAL_EPOCHS - epoch
                eta_seconds = remaining_epochs * SEC_PER_EPOCH
                remaining_models = info["total_models"] - info["current_idx"] - 1
                eta_seconds += remaining_models * TOTAL_EPOCHS * SEC_PER_EPOCH
                eta_hours = eta_seconds / 3600
                display.append(f"  Est. remaining (all models): ~{eta_hours:.1f}h")

            display.append("=" * 60)
            display.append(f"  Updated: {time.strftime('%H:%M:%S')} | Refresh: {interval}s")
            display.append("")

            output = "\n".join(display)

            # Clear previous output using ANSI
            if last_len > 0:
                sys.stdout.write(f"\033[{last_len}A\033[J")
            sys.stdout.write(output)
            sys.stdout.flush()
            last_len = len(display)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
