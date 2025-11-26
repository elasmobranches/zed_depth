"""
Depth Anything v3 모델 평가 스크립트 
ZED depth를 ground truth로 사용하여 상대/절대 깊이 추정 모델을 평가합니다.

개선사항:
- 거리별 silog, spearman 추가
- 거리별 delta_2, delta_3 추가
- 코드 구조 개선
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from tqdm import tqdm
from scipy.stats import spearmanr


class DepthEvaluationFramework:
    """Depth 모델 평가 프레임워크"""
    
    # 클래스 상수
    DISTANCE_RANGES = [
        (0, 1000),      # 0-1m
        (1000, 2000),   # 1-2m
        (2000, 5000),   # 2-5m
        (5000, 10000),  # 5-10m
        (10000, 20000)  # 10-20m
    ]
    
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
            confidence_threshold: ZED confidence threshold (0.0-100.0)
            max_distance: 최대 거리 (mm)
            min_distance: 최소 거리 (mm)
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
        
        # Confidence 디렉토리 확인
        confidence_dir = self.zed_dir.parent / "confidence"
        has_confidence = confidence_dir.exists()
        
        if not has_confidence:
            print("  → Confidence map을 찾을 수 없습니다. Confidence threshold는 적용되지 않습니다.")
        
        zed_depths = []
        zed_confidences = []
        rel_depths = []
        abs_depths = []
        
        for zed_file in tqdm(zed_files, desc="파일 로딩"):
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
            else:
                zed_confidences.append(None)
            
            # 상대 깊이 로드
            rel_file = self.rel_dir / f"{idx}_depth.npy"
            if rel_file.exists():
                rel_depth = np.load(rel_file)
                rel_depths.append(rel_depth)
            else:
                rel_depths.append(None)
                print(f"  ⚠ 상대 깊이 파일 없음: {rel_file}")
            
            # 절대 깊이 로드
            abs_file = self.abs_dir / f"{idx}_depth.npy"
            if abs_file.exists():
                abs_depth = np.load(abs_file)
                abs_depths.append(abs_depth)
            else:
                abs_depths.append(None)
                print(f"  ⚠ 절대 깊이 파일 없음: {abs_file}")
        
        print(f"  ✓ {len(zed_depths)}개 파일 로드 완료")
        
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
        """
        pred_valid = pred_depth[valid_mask].flatten()
        gt_valid = gt_depth[valid_mask].flatten()
        
        if method == 'scale_shift':
            A = np.vstack([pred_valid, np.ones(len(pred_valid))]).T
            b = gt_valid
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            scale, shift = x[0], x[1]
            aligned = scale * pred_depth + shift
        else:
            raise ValueError(f"Invalid method: {method}")
        
        return aligned, scale, shift
    
    # ==================== 메트릭 계산 함수들 ====================
    
    def compute_abs_rel(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """Absolute Relative Error"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        return float(np.mean(np.abs(pred_valid - gt_valid) / (gt_valid + 1e-8)))
    
    def compute_rmse(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """Root Mean Squared Error"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        return float(np.sqrt(np.mean((pred_valid - gt_valid) ** 2)))
    
    def compute_rmse_log(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """Root Mean Squared Error in log space"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        # 음수 방지
        pred_valid = np.maximum(pred_valid, 1e-8)
        gt_valid = np.maximum(gt_valid, 1e-8)
        return float(np.sqrt(np.mean((np.log(pred_valid) - np.log(gt_valid)) ** 2)))
    
    def compute_mae(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
        """Mean Absolute Error"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        return float(np.mean(np.abs(pred_valid - gt_valid)))
    
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
        # 음수 방지
        pred_valid = np.maximum(pred_valid, 1e-8)
        gt_valid = np.maximum(gt_valid, 1e-8)
        ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
        return float(np.mean(ratio < threshold))
    
    def compute_silog(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Optional[float]:
        """Scale-Invariant Logarithmic Error"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        
        if len(pred_valid) < 10:  # 너무 적으면 계산 안 함
            return None
        
        # 음수 방지
        pred_valid = np.maximum(pred_valid, 1e-8)
        gt_valid = np.maximum(gt_valid, 1e-8)
        
        log_diff = np.log(pred_valid) - np.log(gt_valid)
        silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))
        return float(silog)
    
    def compute_spearman_correlation(
        self, 
        pred: np.ndarray, 
        gt: np.ndarray, 
        mask: np.ndarray
    ) -> Optional[float]:
        """Spearman rank correlation"""
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        
        if len(pred_valid) < 10:  # 너무 적으면 계산 안 함
            return None
        
        # 샘플링 (너무 많으면)
        if len(pred_valid) > 100000:
            indices = np.random.choice(len(pred_valid), 100000, replace=False)
            pred_valid = pred_valid[indices]
            gt_valid = gt_valid[indices]
        
        corr, _ = spearmanr(pred_valid, gt_valid)
        return float(corr) if not np.isnan(corr) else None
    
    # ==================== 거리별 분석 ====================
    
    def analyze_by_distance_ranges(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, Dict]:
        """거리별 성능 분석 (모든 메트릭 포함)"""
        results = {}
        
        for min_dist, max_dist in self.DISTANCE_RANGES:
            range_mask = mask & (gt >= min_dist) & (gt < max_dist)
            num_pixels = int(range_mask.sum())
            
            range_key = f"{min_dist/1000:.1f}-{max_dist/1000:.1f}m"
            
            if num_pixels < 100:  # 충분한 픽셀이 없으면 스킵
                continue
            
            # 모든 메트릭 계산
            results[range_key] = {
                # 기본 메트릭
                'abs_rel': self.compute_abs_rel(pred, gt, range_mask),
                'rmse': self.compute_rmse(pred, gt, range_mask),
                'mae': self.compute_mae(pred, gt, range_mask),
                
                # Delta accuracies
                'delta_1': self.compute_delta_accuracy(pred, gt, range_mask, 1.25),
                'delta_2': self.compute_delta_accuracy(pred, gt, range_mask, 1.25**2),
                'delta_3': self.compute_delta_accuracy(pred, gt, range_mask, 1.25**3),
                
                # Scale-invariant 메트릭
                'silog': self.compute_silog(pred, gt, range_mask),
                'spearman': self.compute_spearman_correlation(pred, gt, range_mask),
                
                # 메타 정보
                'num_pixels': num_pixels
            }
        
        return results
    
    # ==================== 모델별 평가 ====================
    
    def evaluate_relative_depth_model(
        self,
        rel_depth: np.ndarray,
        zed_depth: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[Dict, np.ndarray]:
        """상대 깊이 모델 평가"""
        
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
        
        # 1. Scale-invariant 메트릭 (alignment 전, 전체 이미지)
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
        
        # 3. Alignment 후 메트릭 (전체 이미지)
        metrics['after_alignment'] = {
            'abs_rel': self.compute_abs_rel(aligned_rel, zed_depth, valid_mask),
            'rmse': self.compute_rmse(aligned_rel, zed_depth, valid_mask),
            'rmse_log': self.compute_rmse_log(aligned_rel, zed_depth, valid_mask),
            'mae': self.compute_mae(aligned_rel, zed_depth, valid_mask),
            'delta_1': self.compute_delta_accuracy(aligned_rel, zed_depth, valid_mask, 1.25),
            'delta_2': self.compute_delta_accuracy(aligned_rel, zed_depth, valid_mask, 1.25**2),
            'delta_3': self.compute_delta_accuracy(aligned_rel, zed_depth, valid_mask, 1.25**3),
        }
        
        # 4. 거리별 분석 (모든 메트릭 포함)
        metrics['distance_analysis'] = self.analyze_by_distance_ranges(
            aligned_rel, zed_depth, valid_mask
        )
        
        return metrics, aligned_rel
    
    def evaluate_metric_depth_model(
        self,
        abs_depth: np.ndarray,
        zed_depth: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[Dict, np.ndarray]:
        """절대 깊이 모델 평가"""
        
        # 해상도 맞추기
        if abs_depth.shape != zed_depth.shape:
            abs_depth_resized = cv2.resize(
                abs_depth,
                (zed_depth.shape[1], zed_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            abs_depth_resized = abs_depth.copy()
        
        # 단위 변환 (미터 → 밀리미터)
        if abs_depth_resized.max() < 100:  # 미터 단위로 보임
            abs_depth_resized = abs_depth_resized * 1000
        
        metrics = {}
        
        # 1. Scale-invariant 메트릭 (전체 이미지)
        metrics['scale_invariant'] = {
            'spearman': self.compute_spearman_correlation(abs_depth_resized, zed_depth, valid_mask),
            'silog': self.compute_silog(abs_depth_resized, zed_depth, valid_mask),
        }
        
        # 2. 직접 비교 (alignment 없이, 전체 이미지)
        metrics['direct_comparison'] = {
            'abs_rel': self.compute_abs_rel(abs_depth_resized, zed_depth, valid_mask),
            'rmse': self.compute_rmse(abs_depth_resized, zed_depth, valid_mask),
            'rmse_log': self.compute_rmse_log(abs_depth_resized, zed_depth, valid_mask),
            'mae': self.compute_mae(abs_depth_resized, zed_depth, valid_mask),
            'delta_1': self.compute_delta_accuracy(abs_depth_resized, zed_depth, valid_mask, 1.25),
            'delta_2': self.compute_delta_accuracy(abs_depth_resized, zed_depth, valid_mask, 1.25**2),
            'delta_3': self.compute_delta_accuracy(abs_depth_resized, zed_depth, valid_mask, 1.25**3),
        }
        
        # 3. Scale drift 분석
        metrics['scale_drift'] = self.analyze_scale_drift(abs_depth_resized, zed_depth, valid_mask)
        
        # 4. 거리별 분석 (모든 메트릭 포함)
        metrics['distance_analysis'] = self.analyze_by_distance_ranges(
            abs_depth_resized, zed_depth, valid_mask
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
            
            if range_mask.sum() > 100:
                pred_valid = pred[range_mask]
                gt_valid = gt[range_mask]
                
                ratio = gt_valid / (pred_valid + 1e-8)
                scale = np.median(ratio)
                
                scale_factors.append({
                    'distance_range': (float(distance_bins[i]), float(distance_bins[i+1])),
                    'scale_factor': float(scale),
                    'num_pixels': int(range_mask.sum()),
                    'mean_error': float(np.mean(np.abs(pred_valid - gt_valid)))
                })
        
        return scale_factors
    
    # ==================== 전체 평가 ====================
    
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
            zed_confidence = depth_maps['zed_conf'][idx]
            rel_depth = depth_maps['rel'][idx]
            abs_depth = depth_maps['abs'][idx]
            
            if zed_depth is None:
                continue
            
            # Valid mask 생성
            valid_mask = self.create_valid_mask(zed_depth, zed_confidence)
            
            if valid_mask.sum() < 100:
                print(f"  ⚠ 유효 픽셀 부족: {idx}")
                continue
            
            # 상대 깊이 평가
            if rel_depth is not None:
                rel_metrics, _ = self.evaluate_relative_depth_model(
                    rel_depth, zed_depth, valid_mask
                )
                rel_metrics['image_idx'] = idx
                all_rel_metrics.append(rel_metrics)
            
            # 절대 깊이 평가
            if abs_depth is not None:
                abs_metrics, _ = self.evaluate_metric_depth_model(
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
            results['comparison'] = self.compare_models(
                results.get('relative', {}),
                results.get('absolute', {})
            )
        
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
                    if not np.isnan(val) and not np.isinf(val):
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
            if k == 'image_idx':
                continue
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # scale_drift 같은 리스트는 스킵
                continue
            elif isinstance(v, (int, float, np.number)):
                items.append((new_key, float(v)))
            elif v is None:
                # None 값도 저장 (나중에 필터링)
                items.append((new_key, None))
        return dict(items)
    
    def compare_models(self, rel_agg: Dict, abs_agg: Dict) -> Dict:
        """두 모델 비교"""
        
        def get_mean(agg_dict: Dict, key: str) -> Optional[float]:
            if key in agg_dict and isinstance(agg_dict[key], dict):
                return agg_dict[key].get('mean')
            return None
        
        comparison = {
            'summary': {
                'relative_abs_rel': get_mean(rel_agg, 'after_alignment.abs_rel'),
                'absolute_abs_rel': get_mean(abs_agg, 'direct_comparison.abs_rel'),
                'relative_rmse': get_mean(rel_agg, 'after_alignment.rmse'),
                'absolute_rmse': get_mean(abs_agg, 'direct_comparison.rmse'),
                'relative_delta_1': get_mean(rel_agg, 'after_alignment.delta_1'),
                'absolute_delta_1': get_mean(abs_agg, 'direct_comparison.delta_1'),
                'relative_silog': get_mean(rel_agg, 'scale_invariant.silog'),
                'absolute_silog': get_mean(abs_agg, 'scale_invariant.silog'),
                'relative_spearman': get_mean(rel_agg, 'scale_invariant.spearman'),
                'absolute_spearman': get_mean(abs_agg, 'scale_invariant.spearman'),
            }
        }
        
        return comparison
    
    def save_results(self, output_dir: str):
        """결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        def convert_to_serializable(obj: Any) -> Any:
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
        
        print(f"\n✓ 결과 저장: {output_path / 'evaluation_results.json'}")
    
    def print_summary(self):
        """요약 출력"""
        print("\n" + "=" * 60)
        print("평가 결과 요약")
        print("=" * 60)
        
        def get_val(d: Dict, key: str) -> Tuple[Optional[float], Optional[float]]:
            """mean, std 값 가져오기"""
            if key in d and isinstance(d[key], dict):
                return d[key].get('mean'), d[key].get('std')
            return None, None
        
        def fmt(mean: Optional[float], std: Optional[float], suffix: str = '') -> str:
            if mean is None:
                return "N/A"
            if std is not None:
                return f"{mean:.4f} ± {std:.4f}{suffix}"
            return f"{mean:.4f}{suffix}"
        
        def fmt_comp(val: Optional[float], decimals: int = 4, suffix: str = '') -> str:
            """비교용 포맷팅"""
            if val is None:
                return "N/A"
            if decimals == 2:
                return f"{val:.2f}{suffix}"
            return f"{val:.4f}{suffix}"
        
        if 'relative' in self.results:
            print("\n[상대 깊이 모델 (DA3-Mono)]")
            rel = self.results['relative']
            
            m, s = get_val(rel, 'after_alignment.abs_rel')
            print(f"  AbsRel: {fmt(m, s)}")
            
            m, s = get_val(rel, 'after_alignment.rmse')
            print(f"  RMSE: {fmt(m, s, ' mm')}")
            
            m, s = get_val(rel, 'after_alignment.delta_1')
            print(f"  δ1: {fmt(m, s)}")
            
            m, s = get_val(rel, 'scale_invariant.spearman')
            print(f"  Spearman: {fmt(m, s)}")
            
            m, s = get_val(rel, 'scale_invariant.silog')
            print(f"  SILog: {fmt(m, s)}")
        
        if 'absolute' in self.results:
            print("\n[절대 깊이 모델 (DA3-Metric)]")
            abs_ = self.results['absolute']
            
            m, s = get_val(abs_, 'direct_comparison.abs_rel')
            print(f"  AbsRel: {fmt(m, s)}")
            
            m, s = get_val(abs_, 'direct_comparison.rmse')
            print(f"  RMSE: {fmt(m, s, ' mm')}")
            
            m, s = get_val(abs_, 'direct_comparison.delta_1')
            print(f"  δ1: {fmt(m, s)}")
            
            m, s = get_val(abs_, 'scale_invariant.spearman')
            print(f"  Spearman: {fmt(m, s)}")
            
            m, s = get_val(abs_, 'scale_invariant.silog')
            print(f"  SILog: {fmt(m, s)}")
        
        if 'comparison' in self.results:
            print("\n[모델 비교]")
            comp = self.results['comparison']['summary']
            
            rel_abs = fmt_comp(comp.get('relative_abs_rel'))
            abs_abs = fmt_comp(comp.get('absolute_abs_rel'))
            print(f"  AbsRel  - 상대: {rel_abs}, 절대: {abs_abs}")
            
            rel_rmse = fmt_comp(comp.get('relative_rmse'), 2, ' mm')
            abs_rmse = fmt_comp(comp.get('absolute_rmse'), 2, ' mm')
            print(f"  RMSE    - 상대: {rel_rmse}, 절대: {abs_rmse}")
            
            rel_d1 = fmt_comp(comp.get('relative_delta_1'))
            abs_d1 = fmt_comp(comp.get('absolute_delta_1'))
            print(f"  δ1      - 상대: {rel_d1}, 절대: {abs_d1}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Depth 모델 평가")
    parser.add_argument("--zed_dir", type=str, default="./depth_output_zed/move",
                        help="ZED depth 디렉토리")
    parser.add_argument("--rel_dir", type=str, default="./depth_output_rel/move",
                        help="상대 깊이 결과 디렉토리")
    parser.add_argument("--abs_dir", type=str, default="./depth_output_abs/move",
                        help="절대 깊이 결과 디렉토리")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results/move",
                        help="결과 저장 디렉토리")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="ZED confidence threshold (0-100)")
    parser.add_argument("--max_distance", type=float, default=20000.0,
                        help="최대 거리 (mm)")
    parser.add_argument("--min_distance", type=float, default=200.0,
                        help="최소 거리 (mm)")
    
    args = parser.parse_args()
    
    framework = DepthEvaluationFramework(
        zed_dir=args.zed_dir,
        rel_dir=args.rel_dir,
        abs_dir=args.abs_dir,
        confidence_threshold=args.confidence_threshold,
        max_distance=args.max_distance,
        min_distance=args.min_distance
    )
    
    framework.evaluate_all()
    framework.print_summary()
    framework.save_results(args.output_dir)
    
    print("\n✓ 평가 완료!")


if __name__ == "__main__":
    main()