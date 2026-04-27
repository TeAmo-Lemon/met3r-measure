#!/usr/bin/env python3
"""分析 met3r 结果。

1. 差值分析：对每个数据集，找出 abl - StylizedGS 最负的 N 个 style
2. 综合排名：找出跨所有数据集 abl - StylizedGS 差值最低（最负）的 style（用于选测试集）

Usage: python3 analyze_scores.py [N]   (默认 N=10)
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

BASE_DATASETS = ["flower", "garden", "horns", "train", "trex", "truck"]


def load_tsv(path):
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dataset = row["dataset"]
            try:
                short = float(row["short_mean_score"])
                long = float(row["long_mean_score"])
            except (ValueError, KeyError):
                continue
            rows[dataset] = {"short": short, "long": long, "combined": short + long}
    return rows


def compute_diff(abl, stylized):
    common = set(abl) & set(stylized)
    diffs = []
    for ds in common:
        idx = ds.rfind("_")
        base_name = ds[:idx]
        style = ds[idx + 1:]
        diffs.append(
            {
                "base_name": base_name,
                "style": style,
                "abl_short": abl[ds]["short"],
                "abl_long": abl[ds]["long"],
                "abl_combined": abl[ds]["combined"],
                "stl_short": stylized[ds]["short"],
                "stl_long": stylized[ds]["long"],
                "stl_combined": stylized[ds]["combined"],
                "diff_short": abl[ds]["short"] - stylized[ds]["short"],
                "diff_long": abl[ds]["long"] - stylized[ds]["long"],
                "diff_combined": abl[ds]["combined"] - stylized[ds]["combined"],
            }
        )
    return diffs


def print_lowest(title, diffs, key, n):
    grouped = defaultdict(list)
    for d in diffs:
        grouped[d["base_name"]].append(d)

    short_key = key.replace("diff_", "")
    abl_k = "abl_" + short_key
    stl_k = "stl_" + short_key

    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    header = f"{'Base':<14} {'Rank':<6} {'Style':<8} {'Diff':>12} {'abl':>12} {'StylizedGS':>12}"
    print(header)
    print("-" * 100)

    for bn in sorted(grouped.keys()):
        items = grouped[bn]
        items.sort(key=lambda x: x[key])
        for i, d in enumerate(items[:n]):
            print(
                f"{bn:<14} #{i + 1:<5} "
                f"{d['style']:<8} {d[key]:>12.6f} {d[abl_k]:>12.6f} {d[stl_k]:>12.6f}"
            )
        print("-" * 100)


def rank_styles_cross_dataset(diffs):
    """找出在所有6个数据集上都存在的 style，按跨数据集平均 diff 排名。

    返回 {style: {avg_diff_short, avg_diff_long, avg_diff_combined, diff_vals: [...]}}
    """
    # 收集每个 style 在每个 base dataset 上的 diff
    style_diffs = defaultdict(dict)
    style_abl = defaultdict(dict)
    style_stl = defaultdict(dict)
    for d in diffs:
        style_diffs[d["style"]][d["base_name"]] = {
            "diff_short": d["diff_short"],
            "diff_long": d["diff_long"],
            "diff_combined": d["diff_combined"],
        }
        style_abl[d["style"]][d["base_name"]] = {
            "short": d["abl_short"],
            "long": d["abl_long"],
            "combined": d["abl_combined"],
        }
        style_stl[d["style"]][d["base_name"]] = {
            "short": d["stl_short"],
            "long": d["stl_long"],
            "combined": d["stl_combined"],
        }

    complete = {}
    for style, bases in style_diffs.items():
        if set(bases.keys()) == set(BASE_DATASETS):
            diff_combined_vals = [bases[bn]["diff_combined"] for bn in BASE_DATASETS]
            diff_short_vals = [bases[bn]["diff_short"] for bn in BASE_DATASETS]
            diff_long_vals = [bases[bn]["diff_long"] for bn in BASE_DATASETS]
            abl_combined_vals = [style_abl[style][bn]["combined"] for bn in BASE_DATASETS]
            stl_combined_vals = [style_stl[style][bn]["combined"] for bn in BASE_DATASETS]
            complete[style] = {
                "avg_diff_short": sum(diff_short_vals) / len(diff_short_vals),
                "avg_diff_long": sum(diff_long_vals) / len(diff_long_vals),
                "avg_diff_combined": sum(diff_combined_vals) / len(diff_combined_vals),
                "avg_abl_combined": sum(abl_combined_vals) / len(abl_combined_vals),
                "avg_stl_combined": sum(stl_combined_vals) / len(stl_combined_vals),
                "diff_combined_vals": [f"{v:.4f}" for v in diff_combined_vals],
                "abl_combined_vals": [f"{v:.4f}" for v in abl_combined_vals],
                "stl_combined_vals": [f"{v:.4f}" for v in stl_combined_vals],
            }
    return complete


def print_cross_ranking(title, styles, key, n):
    sorted_styles = sorted(styles.items(), key=lambda x: x[1][key])

    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")
    header = (f"{'Rank':<6} {'Style':<8} {'Avg Diff':>10} "
              f"{'Avg abl':>10} {'Avg Stl':>10}  "
              f"{'Per-dataset diff':>65}")
    print(header)
    print("-" * 130)

    for i, (style, s) in enumerate(sorted_styles[:n]):
        detail = " | ".join(
            f"{bn}:{v}"
            for bn, v in zip(BASE_DATASETS, s["diff_combined_vals"])
        )
        print(
            f"#{i + 1:<5} {style:<8} {s[key]:>10.6f} "
            f"{s['avg_abl_combined']:>10.6f} {s['avg_stl_combined']:>10.6f}  {detail}"
        )

    print()
    if sorted_styles:
        print(f"  (共 {len(sorted_styles)} 个 style 在所有 {len(BASE_DATASETS)} 个数据集上都有数据)")
    return [s[0] for s in sorted_styles[:n]]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    base = Path("/home/szw/zxc/met3r/results")

    abl = load_tsv(base / "abl_color_patch.tsv")
    stylized = load_tsv(base / "StylizedGS_met3r.tsv")
    diffs = compute_diff(abl, stylized)

    # ===== Part 1: per-dataset diff =====
    print("\n" + "#" * 120)
    print("#  Part 1: 各数据集 abl - StylizedGS 差值分析（最负 = abl 相对更好）")
    print("#" * 120)

    print_lowest(f"Short diff (abl - StylizedGS), top {n}", diffs, "diff_short", n)
    print_lowest(f"Long diff (abl - StylizedGS), top {n}", diffs, "diff_long", n)
    print_lowest(f"Combined diff (abl - StylizedGS), top {n}", diffs, "diff_combined", n)

    # ===== Part 2: cross-dataset diff ranking =====
    print("\n\n" + "#" * 120)
    print("#  Part 2: 综合排名 — 跨所有数据集 abl - StylizedGS 差值最小的 style")
    print("#" * 120)

    complete = rank_styles_cross_dataset(diffs)

    top_styles = print_cross_ranking(
        f"Combined diff (abl - StylizedGS) 跨数据集均值 最低的 top {n}",
        complete, "avg_diff_combined", n,
    )

    print(f"\n>>> 推荐选取的 style（diff 最低）: {', '.join(top_styles)}")


if __name__ == "__main__":
    main()
