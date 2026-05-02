# 分析 tune_output 的 MET3R 批量评测结果，找出最佳参数组合。
# 用法:
#   python analyze_tune.py                          # 自动查找最新的 tune_output_*.tsv
#   python analyze_tune.py --tsv results/xxx.tsv    # 指定 TSV 文件

import argparse
import csv
import glob
import os
from pathlib import Path
from collections import defaultdict


def extract_label(renders_dir: str) -> str:
    """从 renders_dir 路径中提取参数组合标签 (tune_output/<LABEL>/...)"""
    return Path(renders_dir).parents[3].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tune_output TSV to find best parameter combo.")
    parser.add_argument("--tsv", type=Path, help="Path to TSV results file. If not given, uses the latest.")
    args = parser.parse_args()

    if args.tsv:
        tsv_path = args.tsv
    else:
        results_dir = Path(__file__).resolve().parent / "results"
        tsvs = sorted(glob.glob(str(results_dir / "tune_output_*.tsv")), key=os.path.getmtime)
        if not tsvs:
            print("No tune_output TSV found.")
            return
        tsv_path = Path(tsvs[-1])
        print(f"Using latest TSV: {tsv_path.name}\n")

    # 按 label 聚合
    data: dict[str, list[tuple[float, float]]] = defaultdict(list)

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("short_status") != "success" or row.get("long_status") != "success":
                continue
            label = extract_label(row["renders_dir"])
            short = float(row["short_mean_score"])
            long_ = float(row["long_mean_score"])
            data[label].append((short, long_))

    if not data:
        print("No successful rows found.")
        return

    # 计算每个 label 的平均分并排序
    results = []
    for label, scores in data.items():
        avg_short = sum(s[0] for s in scores) / len(scores)
        avg_long = sum(s[1] for s in scores) / len(scores)
        avg_all = (avg_short + avg_long) / 2
        results.append((label, avg_short, avg_long, avg_all, len(scores)))

    # 按总体平均分排序（越低越好）
    results.sort(key=lambda x: x[3])

    print(f"{'Rank':<5} {'Label':<35} {'Avg Short':<12} {'Avg Long':<12} {'Avg All':<12} {'#Scenes':<8}")
    print("-" * 85)
    for i, (label, avg_s, avg_l, avg_a, count) in enumerate(results, 1):
        print(f"{i:<5} {label:<35} {avg_s:<12.6f} {avg_l:<12.6f} {avg_a:<12.6f} {count:<8}")


if __name__ == "__main__":
    main()
