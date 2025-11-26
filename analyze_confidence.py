#!/usr/bin/env python3
"""
Confidence map 분석 스크립트

예시:
    python analyze_confidence.py \
        --confidence_dir ./depth_output_zed/move/confidence \
        --index 000041 \
        --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_confidence(conf_path: Path) -> np.ndarray:
    if not conf_path.exists():
        raise FileNotFoundError(f"confidence 파일을 찾을 수 없습니다: {conf_path}")
    conf = np.load(conf_path)
    if conf.ndim != 2:
        raise ValueError(f"2D confidence map이 아닙니다: {conf.shape}")
    return conf


def summarize_confidence(conf: np.ndarray) -> dict[str, float]:
    stats = {
        "min": float(conf.min()),
        "max": float(conf.max()),
        "mean": float(conf.mean()),
        "median": float(np.median(conf)),
        "std": float(conf.std()),
        "valid_ratio_thr_0.5": float((conf > 0.5).mean()),
        "valid_ratio_thr_0.75": float((conf > 0.75).mean()),
    }
    return stats


def visualize_confidence(conf: np.ndarray, save_dir: Optional[Path], index: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im = axes[0].imshow(conf, cmap="turbo")
    axes[0].set_title(f"Confidence Map ({index})")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(conf.flatten(), bins=100, color="steelblue", alpha=0.8)
    axes[1].set_title("Confidence Histogram")
    axes[1].set_xlabel("confidence")
    axes[1].set_ylabel("count")

    fig.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{index}_confidence.png"
        fig.savefig(out_path, dpi=200)
        print(f"시각화 저장: {out_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence map 분석")
    parser.add_argument(
        "--confidence_dir",
        type=str,
        required=True,
        help="confidence npy가 위치한 디렉터리",
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="분석할 프레임 인덱스 (예: 000041)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="마스킹 예시용 threshold (기본값: 0.5)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="matplotlib 창에 결과 띄우기 (저장 대신)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./confidence_analysis",
        help="결과 저장 디렉터리 (이미지/JSON)",
    )
    args = parser.parse_args()

    conf_dir = Path(args.confidence_dir)
    conf_path = conf_dir / f"{args.index}.npy"
    conf = load_confidence(conf_path)

    stats = summarize_confidence(conf)
    stats["threshold"] = args.threshold
    stats["valid_ratio_thr_custom"] = float((conf > args.threshold).mean())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{args.index}_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"통계 저장: {json_path}")

    if args.show:
        visualize_confidence(conf, None, args.index)
    else:
        vis_dir = output_dir / "figures"
        visualize_confidence(conf, vis_dir, args.index)


if __name__ == "__main__":
    main()

