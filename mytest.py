import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize

from met3r import MEt3R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MEt3R score on image pairs from one folder.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="raft", choices=["mast3r", "dust3r", "raft"])
    parser.add_argument("--distance", type=str, default="cosine", choices=["cosine", "lpips", "rmse", "psnr", "mse", "ssim"])
    parser.add_argument("--pairing", type=str, default="adjacent", choices=["adjacent", "all_to_first"])
    parser.add_argument("--frame-step", type=int, default=1)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    local_featup_repo = script_dir / "third_party" / "FeatUp"
    if not local_featup_repo.is_dir():
        raise FileNotFoundError(f"Local FeatUp repo not found: {local_featup_repo}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = collect_images(args.input_dir)
    pairs = build_pairs(image_paths, args.pairing, args.frame_step)

    metric = MEt3R(
        img_size=args.img_size,
        use_norm=True,
        backbone=args.backbone,
        feature_backbone="dino16",
        feature_backbone_weights=str(local_featup_repo),
        upsampler="featup",
        distance=args.distance,
        freeze=True,
    ).to(device)

    scores: list[float] = []
    with torch.inference_mode():
        progress = tqdm(pairs, desc="Evaluating", unit="pair", dynamic_ncols=True)
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
    print(f"RESULT pairs={len(scores)} mean_score={mean_score:.6f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()