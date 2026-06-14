from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "camvid"
OUT = ROOT / "template" / "figures"
BENCH = ROOT / "experiments" / "camvid_benchmark" / "outputs"

SAMPLES = [
    "0016E5_07959",
    "0016E5_07961",
    "0016E5_07963",
    "0016E5_07965",
]

METHODS = [
    ("UNetSmall", "UNetSmall"),
    ("TinySegFormer", "TinySegFormer"),
    ("Boundary", "TinySegFormer_Boundary"),
]
VOID_CLASS = 30


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def colorize_label(path: Path, num_classes: int = 32) -> Image.Image:
    import numpy as np

    mask = np.array(Image.open(path), dtype=np.int64)
    rng = np.random.default_rng(123)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[VOID_CLASS] = np.array([0, 0, 0], dtype=np.uint8)
    return Image.fromarray(palette[mask.clip(0, num_classes - 1)])


def save_color_label(path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colorize_label(path).save(out_path)
    return out_path


def draw_panel_grid(
    rows: list[list[tuple[str, Path]]],
    out_path: Path,
    cell_size: tuple[int, int] = (240, 180),
    header_h: int = 34,
    gap: int = 10,
) -> None:
    font = load_font(18)
    cols = max(len(row) for row in rows)
    width = cols * cell_size[0] + (cols + 1) * gap
    height = len(rows) * (cell_size[1] + header_h) + (len(rows) + 1) * gap
    canvas = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)

    for r, row in enumerate(rows):
        y0 = gap + r * (cell_size[1] + header_h + gap)
        for c, (title, path) in enumerate(row):
            x0 = gap + c * (cell_size[0] + gap)
            draw.rounded_rectangle(
                [x0, y0, x0 + cell_size[0], y0 + header_h + cell_size[1]],
                radius=4,
                fill="white",
                outline=(210, 215, 220),
            )
            draw.text((x0 + 8, y0 + 7), title, fill=(30, 35, 40), font=font)
            panel = fit_image(path, cell_size)
            canvas.paste(panel, (x0, y0 + header_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_test_preview() -> None:
    rows = []
    label_dir = OUT / "camvid_color_labels"
    for stem in SAMPLES:
        label = save_color_label(DATA / "labels" / f"{stem}_P.png", label_dir / f"{stem}_gt.png")
        rows.append(
            [
                ("Image", DATA / "images" / f"{stem}.png"),
                ("Ground truth", label),
            ]
        )
    draw_panel_grid(rows, OUT / "camvid_test_preview.png")


def make_prediction_gallery() -> None:
    rows = []
    label_dir = OUT / "camvid_color_labels"
    for stem in SAMPLES:
        label = save_color_label(DATA / "labels" / f"{stem}_P.png", label_dir / f"{stem}_gt.png")
        row: list[tuple[str, Path]] = [
            ("Image", DATA / "images" / f"{stem}.png"),
            ("Ground truth", label),
        ]
        for title, method_dir in METHODS:
            row.append((title, BENCH / method_dir / "visuals" / f"{stem}_pred.png"))
        rows.append(row)
    draw_panel_grid(rows, OUT / "camvid_prediction_gallery.png", cell_size=(200, 150))


def main() -> None:
    make_test_preview()
    make_prediction_gallery()


if __name__ == "__main__":
    main()
