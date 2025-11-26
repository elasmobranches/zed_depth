"""
RGB 이미지에 대한 depth를 저장하는 스크립트
Depth-Anything-3 모델을 사용하여 depth estimation을 수행하고 결과를 저장합니다.
"""

import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
import imageio
from typing import Union, List, Optional

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def save_depth_from_rgb(
    image_path: Union[str, List[str]],
    output_dir_mono: str = "./depth_output_rel",
    output_dir_metric: str = "./depth_output_abs",
    model_name1: str = "depth-anything/da3mono-large",
    model_name2: str = "depth-anything/da3metric-large",
    save_format: str = "both",  # "npy", "png", "both"
    device: Optional[str] = None,
):
    """
    RGB 이미지에 대한 depth를 추정하고 저장합니다.
    
    Args:
        image_path: 입력 이미지 경로 (단일 경로 또는 경로 리스트)
        output_dir: 결과를 저장할 디렉토리
        model_name: 사용할 모델 이름 (Hugging Face Hub)
        save_format: 저장 형식 ("npy", "png", "both")
        device: 사용할 디바이스 ("cuda" 또는 "cpu"), None이면 자동 선택
    
    Returns:
        prediction: DepthAnything3의 Prediction 객체
    """
    # 디바이스 설정
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"사용 디바이스: {device}")
    
    if isinstance(image_path, str):
        image_paths = [image_path]
    else:
        image_paths = image_path
    
    if not image_paths:
        raise ValueError("처리할 이미지가 없습니다.")
    
    model_configs: List[tuple[str, str, Path]] = [
        ("mono", model_name1, Path(output_dir_mono)),
        ("metric", model_name2, Path(output_dir_metric)),
    ]
    last_prediction = None
    
    for tag, model_name, model_output_dir in model_configs:
        print(f"\n[{tag}] 모델 로딩 중: {model_name}")
        model = DepthAnything3.from_pretrained(model_name)
        model = model.to(device=device)
        model.eval()
        print(f"[{tag}] 모델 로딩 완료")
        
        # 출력 디렉토리 설정
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_format in ["png", "both"]:
            vis_dir = model_output_dir / "depth_visualization"
            vis_dir.mkdir(exist_ok=True)
        else:
            vis_dir = None
        
        if save_format in ["npy", "both"]:
            npy_dir = model_output_dir / "depth_npy"
            npy_dir.mkdir(exist_ok=True)
        else:
            npy_dir = None
        
        print(f"[{tag}] {len(image_paths)}개 이미지 처리 중...")
        prediction = model.inference(
            image=image_paths,
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        last_prediction = prediction
        
        print(f"[{tag}] 결과 저장 중...")
        depth_maps = prediction.depth  # shape: (N, H, W)
        
        for idx, img_path in enumerate(image_paths):
            img_name = Path(img_path).stem
            depth_map = depth_maps[idx]
            
            if npy_dir and save_format in ["npy", "both"]:
                npy_path = npy_dir / f"{img_name}_depth.npy"
                np.save(npy_path, depth_map)
                print(f"  [{tag}] NumPy 저장: {npy_path}")
            
            if vis_dir and save_format in ["png", "both"]:
                depth_vis = visualize_depth(depth_map)
                vis_path = vis_dir / f"{img_name}_depth.png"
                imageio.imwrite(vis_path, depth_vis, quality=95)
                print(f"  [{tag}] 시각화 저장: {vis_path}")
        
        print(f"[{tag}] 모든 결과가 {model_output_dir}에 저장되었습니다.")
    
    return last_prediction


def main():
    """메인 함수 - 예제 사용법"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RGB 이미지에 대한 depth를 저장합니다.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="입력 이미지 경로 (단일 이미지 또는 디렉토리)",
    )
    parser.add_argument(
        "--output_mono",
        type=str,
        default="./depth_output_rel/not_move",
        help="모노 모델 출력 디렉토리",
    )
    parser.add_argument(
        "--output_metric",
        type=str,
        default="./depth_output_abs/not_move",
        help="메트릭 모델 출력 디렉토리",
    )
    parser.add_argument(
        "--model_mono",
        type=str,
        default="depth-anything/da3mono-large",
        help="모노 모델",
    )
    parser.add_argument(
        "--model_metric",
        type=str,
        default="depth-anything/da3metric-large",
        help="메트릭 모델",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npy", "png", "both"],
        default="both",
        help="저장 형식 (기본값: both)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="사용할 디바이스 (cuda/cpu, 기본값: 자동 선택)",
    )
    
    args = parser.parse_args()
    
    # 입력 경로 처리
    input_path = Path(args.image)
    if input_path.is_file():
        # 단일 파일
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        # 디렉토리 내의 모든 이미지 파일 찾기 (하위 디렉토리 포함)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_paths = [
            str(p) for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
        if not image_paths:
            print(f"오류: {input_path}에 이미지 파일이 없습니다.")
            return
        image_paths.sort()
    else:
        print(f"오류: {input_path}가 유효한 파일 또는 디렉토리가 아닙니다.")
        return
    
    print(f"처리할 이미지: {len(image_paths)}개")
    output_mono = args.output_mono
    output_metric = args.output_metric
    
    save_depth_from_rgb(
        image_path=image_paths,
        output_dir_mono=output_mono,
        output_dir_metric=output_metric,
        model_name1=args.model_mono,
        model_name2=args.model_metric,
        save_format=args.format,
        device=args.device,
    )


if __name__ == "__main__":
    main()

