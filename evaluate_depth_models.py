"""
Depth Anything v3 모델 평가 스크립트
ZED depth를 ground truth로 사용하여 상대/절대 깊이 추정 모델을 평가합니다.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.optimize import least_squares
import pandas as pd
import json
from tqdm import tqdm


class DepthEvaluationFramework:
    """Depth 모델 평가 프레임워크"""
    
    def __init__(
        self,
        zed_dir: str,
        rel_dir: str,
        abs_dir: str,
        confidence_threshold: float = 0.0,
        max_distance: float = 20000.0,  # mm
        min_distance: float = 200.0,  # mm
    ):
        """
        Args:
            zed_dir: ZED depth npy 파일이 있는 디렉토리
            rel_dir: 상대 깊이 추정 결과 디렉토리
            abs_dir: 절대 깊이 추정 결과 디렉토리
            confidence_threshold: ZED confidence threshold (0.0-1.0)
            max_distance: 최대 거리 (mm)
            min_distance: 최소 거리 (mm)
            최대, 최소 거리는 ZED 2i에서 사용한 설정 값
        """
        self.zed_dir = Path(zed_dir) / "depth_npy"
        self.rel_dir = Path(rel_dir) / "depth_npy"
        self.abs_dir = Path(abs_dir) / "depth_npy"
        
        self.confidence_threshold = confidence_threshold
        self.max_distance = max_distance
        self.min_distance = min_distance
        
        # 결과 저장용
        self.results = {}
        
    def load_depth_maps(self) -> Dict[str, List[np.ndarray]]:
        """모든 depth map과 confidence map 로드"""
        print("Depth maps 로딩 중...")
        
        # 파일 목록 가져오기
        zed_files = sorted(self.zed_dir.glob("*.npy"))
        
        # Confidence 디렉토리 확인 (depth_npy와 같은 레벨에 confidence_npy가 있을 수 있음)
        # Confidence map이 없어도 정상 동작합니다 (confidence_threshold는 무시됨)
        confidence_dir = self.zed_dir.parent / "confidence"
        has_confidence = confidence_dir.exists()
        
        if not has_confidence:
            print("  → Confidence map을 찾을 수 없습니다. Confidence threshold는 적용되지 않습니다.")
        
        zed_depths = []
        zed_confidences = []
        rel_depths = []
        abs_depths = []
        
        for zed_file in tqdm(zed_files):
            # ZED 파일 이름에서 인덱스 추출 (예: 000000.npy -> 000000)
            idx = zed_file.stem
            
            # ZED depth 로드
            zed_depth = np.load(zed_file)
            zed_depths.append(zed_depth)
            
            # ZED confidence 로드
            if has_confidence:
                conf_file = confidence_dir / f"{idx}.npy"
                if conf_file.exists():
                    zed_conf = np.load(conf_file)
                    zed_confidences.append(zed_conf)
                
            else:
                zed_confidences.append(None)
            
            # 상대 깊이 로드
            rel_file = self.rel_dir / f"{idx}_depth.npy"
            if rel_file.exists():
                rel_depth = np.load(rel_file)
                rel_depths.append(rel_depth)
            else:
                rel_depths.append(None)
                print(f"상대 깊이 파일을 찾을 수 없습니다: {rel_file}")
            
            # 절대 깊이 로드
            abs_file = self.abs_dir / f"{idx}_depth.npy"
            if abs_file.exists():
                abs_depth = np.load(abs_file)
                abs_depths.append(abs_depth)
            else:
                abs_depths.append(None)
                print(f"절대 깊이 파일을 찾을 수 없습니다: {abs_file}")
        
        return {
            'zed': zed_depths,
            'zed_conf': zed_confidences,
            'rel': rel_depths,
            'abs': abs_depths
        }
    
    def create_valid_mask(
        self, 
        zed_depth: np.ndarray,
        zed_confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """유효한 픽셀 마스크 생성"""
        mask = (zed_depth > self.min_distance) & (zed_depth < self.max_distance)
        
        if zed_confidence is not None:
            mask = mask & (zed_confidence > self.confidence_threshold)
        
        return mask
    
    def align_depths(
        self,
        pred_depth: np.ndarray,
        gt_depth: np.ndarray,
        valid_mask: np.ndarray,
        method: str = 'scale_shift'
    ) -> Tuple[np.ndarray, float, float]:
        """
        Least squares로 depth alignment 수행
        
        aligned_pred = scale * pred + shift ≈ gt
        상대 깊이와 GT간의 픽셀 관계가 *선형 이라고 가정한다.

        Args:
            pred_depth: 예측 depth
            gt_depth: Ground truth depth
            valid_mask: 유효한 픽셀 마스크
            method: 'scale_shift' 또는 'scale_only'
        
        Returns:
            aligned_depth: pred_depth를 변환한 결과
            scale: 스케일 팩터
            shift: 시프트 값
        """
        pred_valid = pred_depth[valid_mask].flatten()
        gt_valid = gt_depth[valid_mask].flatten()
        
        if method == 'scale_shift':
            # 목표: aligned_pred = scale * pred + shift ≈ gt
            # 최소화: ||scale * pred + shift - gt||^2
            A = np.vstack([pred_valid, np.ones(len(pred_valid))]).T
            b = gt_valid
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            scale, shift = x[0], x[1]
            
            aligned = scale * pred_depth + shift  # pred_depth에 적용
    
        else:
            raise ValueError(f"Invalid method: {method}")
        return aligned, scale, shift
    
    def compute_abs_rel(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """Absolute Relative Error"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        return np.mean(np.abs(pred_valid - gt_valid) / (gt_valid + 1e-8))
    
    def compute_rmse(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """Root Mean Squared Error"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        return np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    
    def compute_rmse_log(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """Root Mean Squared Error in log space"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        return np.sqrt(np.mean((np.log(pred_valid + 1e-8) - np.log(gt_valid + 1e-8)) ** 2))
    
    def compute_delta_accuracy(
        self, 
        pred: np.ndarray, 
        gt: np.ndarray, 
        mask: np.ndarray, 
        threshold: float = 1.25
    ) -> float:
        """Delta accuracy (δ < threshold)"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        ratio = np.maximum(pred_valid / (gt_valid + 1e-8), gt_valid / (pred_valid + 1e-8))
        return np.mean(ratio < threshold)
    
    def compute_spearman_correlation(
        self, 
        pred: np.ndarray, 
        gt: np.ndarray, 
        mask: np.ndarray
    ) -> float:
        """Spearman rank correlation"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        
        # 샘플링 (너무 많으면)
        if len(pred_valid) > 100000:
            indices = np.random.choice(len(pred_valid), 100000, replace=False)
            pred_valid = pred_valid[indices]
            gt_valid = gt_valid[indices]
        
        corr, _ = spearmanr(pred_valid, gt_valid)
        return corr
    
    def compute_silog(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """Scale-Invariant Logarithmic Error"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        
        log_diff = np.log(pred_valid + 1e-8) - np.log(gt_valid + 1e-8)
        return np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))
    
    def compute_mae(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """Mean Absolute Error"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        return np.mean(np.abs(pred_valid - gt_valid))
    
    def analyze_by_distance_ranges(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray,
        ranges: List[Tuple[float, float]]
    ) -> Dict[str, Dict]:
        """거리별 성능 분석"""
        results = {}
        
        for min_dist, max_dist in ranges:
            range_mask = mask & (gt >= min_dist) & (gt < max_dist)
            
            if range_mask.sum() < 100:  # 충분한 픽셀이 없으면 스킵
                continue
            
            results[f"{min_dist/1000:.1f}-{max_dist/1000:.1f}m"] = {
                'abs_rel': self.compute_abs_rel(pred, gt, range_mask),
                'rmse': self.compute_rmse(pred, gt, range_mask),
                'mae': self.compute_mae(pred, gt, range_mask),
                'delta_1': self.compute_delta_accuracy(pred, gt, range_mask, 1.25),
                'num_pixels': range_mask.sum()
            }
        
        return results
    
    def evaluate_relative_depth_model(
        self,
        rel_depth: np.ndarray,
        zed_depth: np.ndarray,
        valid_mask: np.ndarray
    ) -> Dict:
        """상대 깊이 모델 평가"""
        print("상대 깊이 모델 평가 중...")
        
        # 해상도 맞추기
        if rel_depth.shape != zed_depth.shape:
            rel_depth_resized = cv2.resize(
                rel_depth,
                (zed_depth.shape[1], zed_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            rel_depth_resized = rel_depth.copy()
        
        metrics = {}
        
        # 1. Scale-invariant 메트릭 (alignment 전)
        metrics['scale_invariant'] = {
            'spearman': self.compute_spearman_correlation(rel_depth_resized, zed_depth, valid_mask),
            'silog': self.compute_silog(rel_depth_resized, zed_depth, valid_mask),
        }
        
        # 2. Alignment 수행
        aligned_rel, scale, shift = self.align_depths(
            rel_depth_resized, zed_depth, valid_mask, method='scale_shift'
        )
        
        metrics['alignment'] = {
            'scale_factor': float(scale),
            'shift_factor': float(shift)
        }
        
        # 3. Alignment 후 메트릭
        metrics['after_alignment'] = {
            'abs_rel': self.compute_abs_rel(aligned_rel, zed_depth, valid_mask),
            'rmse': self.compute_rmse(aligned_rel, zed_depth, valid_mask),
            'rmse_log': self.compute_rmse_log(aligned_rel, zed_depth, valid_mask),
            'mae': self.compute_mae(aligned_rel, zed_depth, valid_mask),
            'delta_1': self.compute_delta_accuracy(aligned_rel, zed_depth, valid_mask, 1.25),
            'delta_2': self.compute_delta_accuracy(aligned_rel, zed_depth, valid_mask, 1.25**2),
            'delta_3': self.compute_delta_accuracy(aligned_rel, zed_depth, valid_mask, 1.25**3),
        }
        
        # 4. 거리별 분석
        distance_ranges = [
            (0, 1000),      # 0-1m
            (1000, 2000),   # 1-2m
            (2000, 5000),   # 2-5m
            (5000, 10000),  # 5-10m
            (10000, 20000)  # 10-20m
        ]
        metrics['distance_analysis'] = self.analyze_by_distance_ranges(
            aligned_rel, zed_depth, valid_mask, distance_ranges
        )
        
        return metrics, aligned_rel
    
    def evaluate_metric_depth_model(
        self,
        abs_depth: np.ndarray,
        zed_depth: np.ndarray,
        valid_mask: np.ndarray
    ) -> Dict:
        """절대 깊이 모델 평가"""
        print("절대 깊이 모델 평가 중...")
        
        # 해상도 맞추기
        if abs_depth.shape != zed_depth.shape:
            abs_depth_resized = cv2.resize(
                abs_depth,
                (zed_depth.shape[1], zed_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            abs_depth_resized = abs_depth.copy()
        
        # 절대 깊이는 보통 미터 단위이므로 밀리미터로 변환
        # (ZED는 mm 단위라고 가정)
        # 만약 이미 mm 단위라면 변환 불필요
        # 여기서는 값의 범위를 보고 자동 판단
        if abs_depth_resized.max() < 100:  # 미터 단위로 보임
            abs_depth_resized = abs_depth_resized * 1000  # m -> mm
        
        metrics = {}
        
        # 1. Scale-invariant 메트릭 (alignment 없이 직접 비교)
        metrics['scale_invariant'] = {
            'spearman': self.compute_spearman_correlation(abs_depth_resized, zed_depth, valid_mask),
            'silog': self.compute_silog(abs_depth_resized, zed_depth, valid_mask),
        }
        
        # 2. 직접 비교 (alignment 없이)
        metrics['direct_comparison'] = {
            'abs_rel': self.compute_abs_rel(abs_depth_resized, zed_depth, valid_mask),
            'rmse': self.compute_rmse(abs_depth_resized, zed_depth, valid_mask),
            'rmse_log': self.compute_rmse_log(abs_depth_resized, zed_depth, valid_mask),
            'mae': self.compute_mae(abs_depth_resized, zed_depth, valid_mask),
            'delta_1': self.compute_delta_accuracy(abs_depth_resized, zed_depth, valid_mask, 1.25),
            'delta_2': self.compute_delta_accuracy(abs_depth_resized, zed_depth, valid_mask, 1.25**2),
            'delta_3': self.compute_delta_accuracy(abs_depth_resized, zed_depth, valid_mask, 1.25**3),
        }
        
        # 3. Scale drift 분석 (거리에 따른 스케일 변화)
        metrics['scale_drift'] = self.analyze_scale_drift(abs_depth_resized, zed_depth, valid_mask)
        
        # 4. 거리별 분석
        distance_ranges = [
            (0, 1000),      # 0-1m
            (1000, 2000),   # 1-2m
            (2000, 5000),   # 2-5m
            (5000, 10000),  # 5-10m
            (10000, 20000)  # 10-20m
        ]
        metrics['distance_analysis'] = self.analyze_by_distance_ranges(
            abs_depth_resized, zed_depth, valid_mask, distance_ranges
        )
        
        return metrics, abs_depth_resized
    
    def analyze_scale_drift(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray
    ) -> List[Dict]:
        """거리에 따른 스케일 변화 분석"""
        distance_bins = np.linspace(0, 20000, 21)  # 0-20m를 1m 간격으로
        scale_factors = []
        
        for i in range(len(distance_bins) - 1):
            range_mask = mask & (gt >= distance_bins[i]) & (gt < distance_bins[i+1])
            
            if range_mask.sum() > 100:  # 충분한 픽셀이 있을 때만
                pred_valid = pred[range_mask]
                gt_valid = gt[range_mask]
                
                # Median ratio로 스케일 계산
                ratio = gt_valid / (pred_valid + 1e-8)
                scale = np.median(ratio)
                
                scale_factors.append({
                    'distance_range': (float(distance_bins[i]), float(distance_bins[i+1])),
                    'scale_factor': float(scale),
                    'num_pixels': int(range_mask.sum()),
                    'mean_error': float(np.mean(np.abs(pred_valid - gt_valid)))
                })
        
        return scale_factors
    
    def evaluate_all(self) -> Dict:
        """전체 평가 수행"""
        print("=" * 60)
        print("Depth 모델 평가 시작")
        print("=" * 60)
        
        # 데이터 로드
        depth_maps = self.load_depth_maps()
        
        all_rel_metrics = []
        all_abs_metrics = []
        
        # 각 이미지에 대해 평가
        for idx in tqdm(range(len(depth_maps['zed'])), desc="이미지 평가"):
            zed_depth = depth_maps['zed'][idx]
            zed_confidence = depth_maps.get('zed_conf', [None] * len(depth_maps['zed']))[idx]
            rel_depth = depth_maps['rel'][idx]
            abs_depth = depth_maps['abs'][idx]

            if zed_depth is None:
                print(f"ZED depth 파일을 찾을 수 없습니다: {idx}")
                continue
            
            # Valid mask 생성 (confidence 포함)
            valid_mask = self.create_valid_mask(zed_depth, zed_confidence)
            
            if valid_mask.sum() < 100:  # 유효한 픽셀이 너무 적으면 스킵
                print(f"유효한 픽셀이 너무 적습니다: {idx}")
                continue
            
            # 상대 깊이 평가
            if rel_depth is not None:
                rel_metrics, aligned_rel = self.evaluate_relative_depth_model(
                    rel_depth, zed_depth, valid_mask
                )
                rel_metrics['image_idx'] = idx
                all_rel_metrics.append(rel_metrics)
            
            # 절대 깊이 평가
            if abs_depth is not None:
                abs_metrics, aligned_abs = self.evaluate_metric_depth_model(
                    abs_depth, zed_depth, valid_mask
                )
                abs_metrics['image_idx'] = idx
                all_abs_metrics.append(abs_metrics)
        
        # 전체 평균 계산
        results = {}
        
        if all_rel_metrics:
            results['relative'] = self.aggregate_metrics(all_rel_metrics)
        
        if all_abs_metrics:
            results['absolute'] = self.aggregate_metrics(all_abs_metrics)
        
        # 비교 분석
        if all_rel_metrics and all_abs_metrics:
            results['comparison'] = self.compare_models(all_rel_metrics, all_abs_metrics)
        
        self.results = results
        return results
    
    def aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """여러 이미지의 메트릭을 평균"""
        if not metrics_list:
            return {}
        
        # 평탄화된 딕셔너리 리스트
        flattened_list = [self._flatten_dict(m) for m in metrics_list]
        
        # 모든 키 수집
        all_keys = set()
        for flat_dict in flattened_list:
            all_keys.update(flat_dict.keys())
        
        # 평균 계산
        aggregated = {}
        for key in all_keys:
            values = []
            for flat_dict in flattened_list:
                val = flat_dict.get(key)
                if val is not None and isinstance(val, (int, float, np.number)):
                    values.append(float(val))
            
            if values:
                aggregated[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return aggregated
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """딕셔너리를 평탄화"""
        items = []
        for k, v in d.items():
            if k == 'image_idx':  # image_idx는 제외
                continue
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, np.number)):
                items.append((new_key, float(v)))
        return dict(items)
    
    def compare_models(self, rel_metrics: List[Dict], abs_metrics: List[Dict]) -> Dict:
        """두 모델 비교"""
        comparison = {}
        
        # 주요 메트릭 비교
        rel_agg = self.aggregate_metrics(rel_metrics)
        abs_agg = self.aggregate_metrics(abs_metrics)
        
        def get_nested_mean(agg_dict, key_path):
            """중첩된 딕셔너리에서 mean 값 가져오기
            JSON이 평탄화되어 있으므로 (예: "after_alignment.abs_rel": {"mean": ...})
            key_path가 "after_alignment.abs_rel" 형태일 때 올바르게 처리
            """
            # 직접 키로 찾기 (평탄화된 구조)
            if key_path in agg_dict:
                val = agg_dict[key_path]
                if isinstance(val, dict) and 'mean' in val:
                    return val['mean']
            
            # 중첩된 구조 시도
            keys = key_path.split('.')
            val = agg_dict
            for k in keys:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    return None
            if isinstance(val, dict) and 'mean' in val:
                return val['mean']
            return None
        
        comparison['summary'] = {
            'relative_abs_rel': get_nested_mean(rel_agg, 'after_alignment.abs_rel'),
            'absolute_abs_rel': get_nested_mean(abs_agg, 'direct_comparison.abs_rel'),
            'relative_rmse': get_nested_mean(rel_agg, 'after_alignment.rmse'),
            'absolute_rmse': get_nested_mean(abs_agg, 'direct_comparison.rmse'),
            'relative_delta_1': get_nested_mean(rel_agg, 'after_alignment.delta_1'),
            'absolute_delta_1': get_nested_mean(abs_agg, 'direct_comparison.delta_1'),
        }
        
        return comparison
    
    def save_results(self, output_dir: str):
        """결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSON으로 저장
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        with open(output_path / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(self.results), f, indent=2, ensure_ascii=False)
        
        print(f"\n결과가 {output_path / 'evaluation_results.json'}에 저장되었습니다.")
    
    def print_summary(self):
        """요약 출력"""
        print("\n" + "=" * 60)
        print("평가 결과 요약")
        print("=" * 60)
        
        def get_nested_value(d, key_path, default=None):
            """중첩된 딕셔너리에서 값 가져오기"""
            keys = key_path.split('.')
            val = d
            for k in keys:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    return default
            return val
        
        if 'relative' in self.results:
            print("\n[상대 깊이 모델 (DA3-Mono)]")
            rel = self.results['relative']
            
            abs_rel_mean = get_nested_value(rel, 'after_alignment.abs_rel.mean')
            abs_rel_std = get_nested_value(rel, 'after_alignment.abs_rel.std')
            if abs_rel_mean is not None:
                print(f"  AbsRel: {abs_rel_mean:.4f} ± {abs_rel_std:.4f}" if abs_rel_std is not None else f"  AbsRel: {abs_rel_mean:.4f}")
            
            rmse_mean = get_nested_value(rel, 'after_alignment.rmse.mean')
            rmse_std = get_nested_value(rel, 'after_alignment.rmse.std')
            if rmse_mean is not None:
                print(f"  RMSE: {rmse_mean:.2f} ± {rmse_std:.2f} mm" if rmse_std is not None else f"  RMSE: {rmse_mean:.2f} mm")
            
            delta_mean = get_nested_value(rel, 'after_alignment.delta_1.mean')
            delta_std = get_nested_value(rel, 'after_alignment.delta_1.std')
            if delta_mean is not None:
                print(f"  δ1: {delta_mean:.4f} ± {delta_std:.4f}" if delta_std is not None else f"  δ1: {delta_mean:.4f}")
            
            spearman_mean = get_nested_value(rel, 'scale_invariant.spearman.mean')
            spearman_std = get_nested_value(rel, 'scale_invariant.spearman.std')
            if spearman_mean is not None:
                print(f"  Spearman: {spearman_mean:.4f} ± {spearman_std:.4f}" if spearman_std is not None else f"  Spearman: {spearman_mean:.4f}")
            
            silog_mean = get_nested_value(rel, 'scale_invariant.silog.mean')
            silog_std = get_nested_value(rel, 'scale_invariant.silog.std')
            if silog_mean is not None:
                print(f"  SILog: {silog_mean:.4f} ± {silog_std:.4f}" if silog_std is not None else f"  SILog: {silog_mean:.4f}")
        
        if 'absolute' in self.results:
            print("\n[절대 깊이 모델 (DA3-Metric)]")
            abs_ = self.results['absolute']
            
            abs_rel_mean = get_nested_value(abs_, 'direct_comparison.abs_rel.mean')
            abs_rel_std = get_nested_value(abs_, 'direct_comparison.abs_rel.std')
            if abs_rel_mean is not None:
                print(f"  AbsRel: {abs_rel_mean:.4f} ± {abs_rel_std:.4f}" if abs_rel_std is not None else f"  AbsRel: {abs_rel_mean:.4f}")
            
            rmse_mean = get_nested_value(abs_, 'direct_comparison.rmse.mean')
            rmse_std = get_nested_value(abs_, 'direct_comparison.rmse.std')
            if rmse_mean is not None:
                print(f"  RMSE: {rmse_mean:.2f} ± {rmse_std:.2f} mm" if rmse_std is not None else f"  RMSE: {rmse_mean:.2f} mm")
            
            delta_mean = get_nested_value(abs_, 'direct_comparison.delta_1.mean')
            delta_std = get_nested_value(abs_, 'direct_comparison.delta_1.std')
            if delta_mean is not None:
                print(f"  δ1: {delta_mean:.4f} ± {delta_std:.4f}" if delta_std is not None else f"  δ1: {delta_mean:.4f}")
        
        if 'comparison' in self.results:
            print("\n[모델 비교]")
            comp = self.results['comparison']['summary']
            
            def format_value(val):
                """값을 안전하게 포맷"""
                if val is None:
                    return "N/A"
                try:
                    return f"{val:.4f}"
                except (TypeError, ValueError):
                    return str(val)
            
            rel_abs_rel = comp.get('relative_abs_rel')
            abs_abs_rel = comp.get('absolute_abs_rel')
            rel_rmse = comp.get('relative_rmse')
            abs_rmse = comp.get('absolute_rmse')
            rel_delta = comp.get('relative_delta_1')
            abs_delta = comp.get('absolute_delta_1')
            
            if rel_abs_rel is not None or abs_abs_rel is not None:
                print(f"  상대 모델 AbsRel: {format_value(rel_abs_rel)}")
                print(f"  절대 모델 AbsRel: {format_value(abs_abs_rel)}")
            if rel_rmse is not None or abs_rmse is not None:
                print(f"  상대 모델 RMSE: {format_value(rel_rmse)} mm")
                print(f"  절대 모델 RMSE: {format_value(abs_rmse)} mm")
            if rel_delta is not None or abs_delta is not None:
                print(f"  상대 모델 δ1: {format_value(rel_delta)}")
                print(f"  절대 모델 δ1: {format_value(abs_delta)}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Depth 모델 평가")
    parser.add_argument(
        "--zed_dir",
        type=str,
        default="./depth_output_zed/not_move",
        help="ZED depth 디렉토리"
    )
    parser.add_argument(
        "--rel_dir",
        type=str,
        default="./depth_output_rel/not_move",
        help="상대 깊이 결과 디렉토리"
    )
    parser.add_argument(
        "--abs_dir",
        type=str,
        default="./depth_output_abs/not_move",
        help="절대 깊이 결과 디렉토리"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results/not_move",
        help="결과 저장 디렉토리"
    )
    # 기본값은 __init__의 기본값과 동일하게 유지 (일관성 유지)
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.0,  # __init__의 기본값과 동일
        help="ZED confidence threshold (0-100, 기본값: 0.0)"
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=20000.0,  # __init__의 기본값과 동일
        help="최대 거리 (mm, 기본값: 20000.0)"
    )
    parser.add_argument(
        "--min_distance",
        type=float,
        default=200.0,  # __init__의 기본값과 동일
        help="최소 거리 (mm, 기본값: 200.0)"
    )
    
    args = parser.parse_args()
    
    # 평가 프레임워크 초기화
    # 명령줄 인자가 제공되면 args.* 사용, 없으면 argparse의 default 사용
    framework = DepthEvaluationFramework(
        zed_dir=args.zed_dir,
        rel_dir=args.rel_dir,
        abs_dir=args.abs_dir,
        confidence_threshold=args.confidence_threshold,
        max_distance=args.max_distance,
        min_distance=args.min_distance
    )
    
    # 평가 실행
    framework.evaluate_all()
    
    # 결과 출력
    framework.print_summary()
    framework.save_results(args.output_dir)
    print("\n평가 완료!")


if __name__ == "__main__":
    main()

