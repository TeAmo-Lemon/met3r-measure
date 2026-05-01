import argparse
import csv
import os
import re
import glob
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize

from met3r import MEt3R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MEt3R score on image pairs from one folder.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--batch-root-dir",
        type=Path,
        help="Root directory to batch-scan folders matching */train/ours_*/renders and save TSV.",
    )
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="raft", choices=["mast3r", "dust3r", "raft"])
    parser.add_argument("--distance", type=str, default="cosine", choices=["cosine", "lpips", "rmse", "psnr", "mse", "ssim"])
    parser.add_argument("--pairing", type=str, default="adjacent", choices=["adjacent", "all_to_first"])
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--short-step", type=int, default=1)
    parser.add_argument("--long-step", type=int, default=5)
    args = parser.parse_args()
    if args.input_dir is None and args.batch_root_dir is None:
        parser.error("Provide one of --input-dir or --batch-root-dir")
    if args.short_step < 1 or args.long_step < 1:
        parser.error("--short-step and --long-step must be >= 1")
    return args


def load_rgb_tensor(image_path: Path, img_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = resize(image, [img_size, img_size], interpolation=InterpolationMode.BICUBIC)
    tensor = pil_to_tensor(image).float() / 255.0
    return (tensor * 2.0 - 1.0).to(device)


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if len(images) < 2:
        raise ValueError(f"Need at least 2 images in {input_dir}, but found {len(images)}")
    return images


def build_pairs(images: list[Path], pairing: str, frame_step: int) -> list[tuple[Path, Path]]:
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, but got {frame_step}")
    if frame_step >= len(images):
        raise ValueError(f"frame_step ({frame_step}) is too large for {len(images)} images")

    if pairing == "adjacent":
        return [(images[i], images[i + frame_step]) for i in range(len(images) - frame_step)]
    return [(images[0], images[i]) for i in range(frame_step, len(images), frame_step)]


def build_metric(args: argparse.Namespace, local_featup_repo: Path, device: torch.device) -> MEt3R:
    return MEt3R(
        img_size=args.img_size,
        use_norm=True,
        backbone=args.backbone,
        feature_backbone="dino16",
        feature_backbone_weights=str(local_featup_repo),
        upsampler="featup",
        distance=args.distance,
        freeze=True,
    ).to(device)


def evaluate_input_dir(
    input_dir: Path,
    args: argparse.Namespace,
    metric: MEt3R,
    device: torch.device,
    frame_step: int,
) -> tuple[int, float]:
    image_paths = collect_images(input_dir)
    pairs = build_pairs(image_paths, args.pairing, frame_step)

    scores: list[float] = []
    with torch.inference_mode():
        progress = tqdm(pairs, desc=f"Evaluating {input_dir.name}", unit="pair", dynamic_ncols=True, leave=False)
        for image1_path, image2_path in progress:
            image1 = load_rgb_tensor(image1_path, args.img_size, device)
            image2 = load_rgb_tensor(image2_path, args.img_size, device)

            inputs = torch.stack([image1, image2], dim=0).unsqueeze(0)
            score, *_ = metric(
                images=inputs,
                return_overlap_mask=False,
                return_score_map=False,
                return_projections=False,
            )
            score_value = score.mean().item()
            scores.append(score_value)
            progress.set_postfix(score=f"{score_value:.4f}")

    mean_score = sum(scores) / len(scores)
    return len(scores), mean_score


def collect_render_dirs(batch_root_dir: Path) -> list[Path]:
    render_dirs: list[Path] = []
    for path in batch_root_dir.rglob("renders"):
        if not path.is_dir():
            continue
        if path.parent.name.startswith("ours_") and path.parent.parent.name == "train":
            render_dirs.append(path)

    def natural_key(text: str) -> list[object]:
        return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]

    render_dirs.sort(key=lambda p: (natural_key(p.parents[2].name), natural_key(str(p))))
    return render_dirs


def get_cached_records(results_dir: Path) -> dict[str, dict]:
    """获取最新 TSV 中已经成功的记录"""
    tsvs = glob.glob(str(results_dir / "met3r_batch_results_*.tsv"))
    if not tsvs:
        return {}
    
    latest_tsv = Path(max(tsvs, key=os.path.getmtime))
    print(f"[INFO] Found previous results: {latest_tsv.name}, loading successful records...")
    
    cached = {}
    try:
        with open(latest_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # 只有当 short 和 long 都成功时，才认为该文件夹处理完毕
                if row.get("short_status") == "success" and row.get("long_status") == "success":
                    cached[row["renders_dir"]] = row
    except Exception as e:
        print(f"[WARN] Failed to parse latest TSV: {e}")
    return cached


def run_batch(batch_root_dir: Path, args: argparse.Namespace, metric: MEt3R, device: torch.device) -> Path:
    if not batch_root_dir.exists() or not batch_root_dir.is_dir():
        raise NotADirectoryError(f"Batch root directory not found: {batch_root_dir}")

    render_dirs = collect_render_dirs(batch_root_dir)
    if not render_dirs:
        raise FileNotFoundError(f"No directories matched */train/ours_*/renders under: {batch_root_dir}")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载已存在的成功记录
    cached_records = get_cached_records(results_dir)

    out_file = results_dir / f"{batch_root_dir.name}_{datetime.now():%Y%m%d_%H%M%S}.tsv"
    
    print(f"[INFO] Batch root directory: {batch_root_dir}")
    print(f"[INFO] Found {len(render_dirs)} render directories ({len(cached_records)} may be skipped)")
    print(f"[INFO] Output TSV: {out_file}")

    fieldnames = [
        "dataset", "short_step", "short_pairs", "short_mean_score", "short_status",
        "long_step", "long_pairs", "long_mean_score", "long_status", "renders_dir", "message"
    ]

    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        f.flush()

        for render_dir in render_dirs:
            dataset = render_dir.parents[2].name if len(render_dir.parents) >= 3 else render_dir.name
            dir_str = str(render_dir)

            # 断点续传检查
            if dir_str in cached_records:
                print(f"[SKIP] {dataset} -> Reusing cached success result.")
                writer.writerow(cached_records[dir_str])
                f.flush()
                continue

            print(f"[RUN] {dataset} -> {render_dir}")

            # --- Short Step Evaluation ---
            short_status, short_pairs, short_mean_score, short_msg = "success", "NA", "NA", ""
            try:
                p, m = evaluate_input_dir(render_dir, args, metric, device, frame_step=args.short_step)
                short_pairs, short_mean_score = str(p), f"{m:.6f}"
            except Exception as exc:
                short_status, short_msg = "failed", f"short: {exc}"
                print(f"[WARN] {dataset} short failed: {exc}")

            # --- Long Step Evaluation ---
            long_status, long_pairs, long_mean_score, long_msg = "success", "NA", "NA", ""
            try:
                p, m = evaluate_input_dir(render_dir, args, metric, device, frame_step=args.long_step)
                long_pairs, long_mean_score = str(p), f"{m:.6f}"
            except Exception as exc:
                long_status, long_msg = "failed", f"long: {exc}"
                print(f"[WARN] {dataset} long failed: {exc}")

            # --- Write Row ---
            writer.writerow({
                "dataset": dataset,
                "short_step": args.short_step,
                "short_pairs": short_pairs,
                "short_mean_score": short_mean_score,
                "short_status": short_status,
                "long_step": args.long_step,
                "long_pairs": long_pairs,
                "long_mean_score": long_mean_score,
                "long_status": long_status,
                "renders_dir": dir_str,
                "message": "; ".join([m for m in [short_msg, long_msg] if m]),
            })
            f.flush()
            
    return out_file


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    local_featup_repo = script_dir / "third_party" / "FeatUp"
    if not local_featup_repo.is_dir():
        raise FileNotFoundError(f"Local FeatUp repo not found: {local_featup_repo}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = build_metric(args, local_featup_repo, device)

    if args.batch_root_dir is not None:
        out_file = run_batch(args.batch_root_dir, args, metric, device)
        print(f"[DONE] Batch evaluation finished: {out_file}")
    else:
        pairs, mean_score = evaluate_input_dir(args.input_dir, args, metric, device, frame_step=args.frame_step)
        print(f"RESULT pairs={pairs} mean_score={mean_score:.6f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()