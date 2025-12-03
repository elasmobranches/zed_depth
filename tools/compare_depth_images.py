"""
두 개의 Depth Map (numpy 파일) 비교 도구

사용 예시:
    python compare_depth_images.py \
        --pred1 depth_output_rel/origin/depth_npy/000001.npy \
        --pred2 depth_output_abs/origin/depth_npy/000001.npy \
        --gt depth_output_zed/origin/depth_npy/000001.npy \
        --name1 "Monocular" \
        --name2 "Metric" \
        --output comparison_results

기능:
- 두 예측 depth map을 GT와 비교
- 모든 metric 계산 (AbsRel, RMSE, MAE, δ1/δ2/δ3, Spearman, SILog)
- 거리별 분석
- 차이 시각화
- 상세한 비교 리포트 생성
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
from scipy.stats import spearmanr
import argparse
import csv
import cv2


class DepthComparator:
    """두 Depth Map 비교 클래스"""
    
    # 거리 범위 (mm)
    DISTANCE_RANGES = [
        (0, 1000, '0.0-1.0m'),
        (1000, 2000, '1.0-2.0m'),
        (2000, 5000, '2.0-5.0m'),
        (5000, 10000, '5.0-10.0m'),
        (10000, 20000, '10.0-20.0m')
    ]
    
    def __init__(
        self,
        pred1_path: str,
        pred2_path: str,
        gt_path: str,
        name1: str = "Pred1",
        name2: str = "Pred2",
        confidence_path: Optional[str] = None,
        confidence_threshold: float = 30.0,
        max_distance: float = 20000.0,
        min_distance: float = 200.0,
        output_dir: str = "comparison_results"
    ):
        """
        Args:
            pred1_path: 첫 번째 예측 depth map 경로 (.npy)
            pred2_path: 두 번째 예측 depth map 경로 (.npy)
            gt_path: Ground truth depth map 경로 (.npy)
            name1: 첫 번째 모델 이름
            name2: 두 번째 모델 이름
            confidence_path: Confidence map 경로 (선택)
            confidence_threshold: Confidence threshold (0-100)
            max_distance: 최대 거리 (mm)
            min_distance: 최소 거리 (mm)
            output_dir: 결과 저장 디렉토리
        """
        self.pred1 = np.load(pred1_path)
        self.pred2 = np.load(pred2_path)
        self.gt = np.load(gt_path)
        
        # 이미지 크기 맞추기 (GT 크기에 맞춤)
        gt_height, gt_width = self.gt.shape
        
        if self.pred1.shape != self.gt.shape:
            print(f"  ⚠ Pred1 크기 조정: {self.pred1.shape} -> {self.gt.shape}")
            self.pred1 = cv2.resize(self.pred1, (gt_width, gt_height), interpolation=cv2.INTER_LINEAR)
        
        if self.pred2.shape != self.gt.shape:
            print(f"  ⚠ Pred2 크기 조정: {self.pred2.shape} -> {self.gt.shape}")
            self.pred2 = cv2.resize(self.pred2, (gt_width, gt_height), interpolation=cv2.INTER_LINEAR)
        
        self.name1 = name1
        self.name2 = name2
        
        # Confidence map 로드
        if confidence_path and Path(confidence_path).exists():
            self.confidence = np.load(confidence_path)
        else:
            self.confidence = None
        
        self.confidence_threshold = confidence_threshold
        self.max_distance = max_distance
        self.min_distance = min_distance
        
        # 출력 디렉토리
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 결과 저장
        self.results = {}
        
        # 스타일 설정
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """플롯 스타일 설정"""
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['figure.dpi'] = 100
    
    def _create_valid_mask(self, pred: np.ndarray) -> np.ndarray:
        """유효한 픽셀 마스크 생성"""
        # GT 유효 범위
        mask = (self.gt > self.min_distance) & (self.gt < self.max_distance)
        
        # Prediction 유효 체크
        mask &= (pred > 0) & np.isfinite(pred)
        
        # Confidence threshold
        if self.confidence is not None:
            mask &= (self.confidence >= self.confidence_threshold)
        
        return mask
    
    def _compute_scale_and_shift(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        """Least squares를 사용한 scale과 shift 계산"""
        pred_valid = pred[mask].flatten()
        gt_valid = gt[mask].flatten()
        
        if len(pred_valid) < 10:
            return 1.0, 0.0
        
        # pred * scale + shift = gt
        # [pred, 1] @ [scale, shift] = gt
        A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
        scale, shift = np.linalg.lstsq(A, gt_valid, rcond=None)[0]
        
        return float(scale), float(shift)
    
    def _compute_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray,
        align_scale: bool = False
    ) -> Dict[str, float]:
        """메트릭 계산"""
        pred_valid = pred[mask].flatten()
        gt_valid = gt[mask].flatten()
        
        if len(pred_valid) == 0:
            return {
                'abs_rel': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'delta_1': np.nan,
                'delta_2': np.nan,
                'delta_3': np.nan,
                'spearman': np.nan,
                'silog': np.nan,
                'num_pixels': 0
            }
        
        # Scale alignment (Monocular 모델용)
        if align_scale:
            scale, shift = self._compute_scale_and_shift(pred, gt, mask)
            pred_aligned = pred * scale + shift
            pred_valid = pred_aligned[mask].flatten()
        
        # AbsRel
        abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
        
        # MAE
        mae = np.mean(np.abs(pred_valid - gt_valid))
        
        # Delta thresholds
        thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
        delta_1 = np.mean(thresh < 1.25)
        delta_2 = np.mean(thresh < 1.25 ** 2)
        delta_3 = np.mean(thresh < 1.25 ** 3)
        
        # Spearman correlation
        if len(pred_valid) > 10:
            spearman, _ = spearmanr(pred_valid, gt_valid)
        else:
            spearman = np.nan
        
        # Scale-Invariant Log error
        log_diff = np.log(pred_valid) - np.log(gt_valid)
        silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))
        
        return {
            'abs_rel': float(abs_rel),
            'rmse': float(rmse),
            'mae': float(mae),
            'delta_1': float(delta_1),
            'delta_2': float(delta_2),
            'delta_3': float(delta_3),
            'spearman': float(spearman),
            'silog': float(silog),
            'num_pixels': int(np.sum(mask))
        }
    
    def _compute_distance_metrics(
        self,
        pred: np.ndarray,
        mask: np.ndarray,
        align_scale: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """거리별 메트릭 계산"""
        results = {}
        
        for min_dist, max_dist, label in self.DISTANCE_RANGES:
            # 해당 거리 범위 마스크
            dist_mask = mask & (self.gt >= min_dist) & (self.gt < max_dist)
            
            if np.sum(dist_mask) < 10:
                continue
            
            metrics = self._compute_metrics(pred, self.gt, dist_mask, align_scale)
            results[label] = metrics
        
        return results
    
    def compare(self):
        """두 depth map 비교"""
        print("\n" + "=" * 60)
        print(f"Depth Map 비교: {self.name1} vs {self.name2}")
        print("=" * 60 + "\n")
        
        # 1. 전체 메트릭 계산
        print("1. 전체 메트릭 계산 중...")
        
        # Pred1 평가 (Monocular 가정 - scale alignment 적용)
        mask1 = self._create_valid_mask(self.pred1)
        metrics1_aligned = self._compute_metrics(self.pred1, self.gt, mask1, align_scale=True)
        metrics1_direct = self._compute_metrics(self.pred1, self.gt, mask1, align_scale=False)
        
        # Pred2 평가 (Metric 가정 - direct comparison)
        mask2 = self._create_valid_mask(self.pred2)
        metrics2_direct = self._compute_metrics(self.pred2, self.gt, mask2, align_scale=False)
        metrics2_aligned = self._compute_metrics(self.pred2, self.gt, mask2, align_scale=True)
        
        self.results['overall'] = {
            self.name1: {
                'with_alignment': metrics1_aligned,
                'without_alignment': metrics1_direct
            },
            self.name2: {
                'with_alignment': metrics2_aligned,
                'without_alignment': metrics2_direct
            }
        }
        
        # 2. 거리별 메트릭 계산
        print("2. 거리별 메트릭 계산 중...")
        
        dist_metrics1 = self._compute_distance_metrics(self.pred1, mask1, align_scale=True)
        dist_metrics2 = self._compute_distance_metrics(self.pred2, mask2, align_scale=False)
        
        self.results['distance_analysis'] = {
            self.name1: dist_metrics1,
            self.name2: dist_metrics2
        }
        
        print("✓ 메트릭 계산 완료\n")
    
    def visualize(self):
        """비교 결과 시각화"""
        print("3. 시각화 생성 중...")
        
        # 1. Depth map 비교 시각화
        self._plot_depth_maps()
        
        # 2. 전체 메트릭 비교
        self._plot_overall_metrics()
        
        # 3. 거리별 메트릭 비교
        self._plot_distance_metrics()
        
        # 4. 에러 맵 시각화
        self._plot_error_maps()
        
        print("✓ 시각화 완료\n")
    
    def _plot_depth_maps(self):
        """Depth map 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Valid masks
        mask1 = self._create_valid_mask(self.pred1)
        mask2 = self._create_valid_mask(self.pred2)
        
        # Scale alignment for pred1
        scale1, shift1 = self._compute_scale_and_shift(self.pred1, self.gt, mask1)
        pred1_aligned = self.pred1 * scale1 + shift1
        
        # Visualization range
        vmin = np.percentile(self.gt[mask1 | mask2], 5)
        vmax = np.percentile(self.gt[mask1 | mask2], 95)
        
        # GT
        im0 = axes[0, 0].imshow(self.gt, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Ground Truth (ZED)', fontsize=12)
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Depth (mm)')
        
        # Pred1 (aligned)
        im1 = axes[0, 1].imshow(pred1_aligned, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f'{self.name1} (Scale Aligned)', fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label='Depth (mm)')
        
        # Pred2
        im2 = axes[1, 0].imshow(self.pred2, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f'{self.name2}', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Depth (mm)')
        
        # Difference map (Pred1 vs Pred2)
        diff = np.abs(pred1_aligned - self.pred2)
        diff_masked = np.ma.masked_where(~(mask1 & mask2), diff)
        im3 = axes[1, 1].imshow(diff_masked, cmap='RdYlGn_r', vmin=0, vmax=np.percentile(diff[mask1 & mask2], 95))
        axes[1, 1].set_title(f'Absolute Difference: {self.name1} vs {self.name2}', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label='|Diff| (mm)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'depth_maps_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Depth maps 비교: {self.output_dir / 'depth_maps_comparison.png'}")
        plt.close()
    
    def _plot_overall_metrics(self):
        """전체 메트릭 비교 차트"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 데이터 추출
        m1 = self.results['overall'][self.name1]['with_alignment']
        m2 = self.results['overall'][self.name2]['without_alignment']
        
        models = [self.name1, self.name2]
        colors = ['steelblue', 'coral']
        
        # 1. AbsRel
        axes[0, 0].bar(models, [m1['abs_rel'], m2['abs_rel']], color=colors, alpha=0.8)
        axes[0, 0].set_ylabel('AbsRel')
        axes[0, 0].set_title('Absolute Relative Error', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. RMSE
        axes[0, 1].bar(models, [m1['rmse'], m2['rmse']], color=colors, alpha=0.8)
        axes[0, 1].set_ylabel('RMSE (mm)')
        axes[0, 1].set_title('Root Mean Square Error', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. MAE
        axes[0, 2].bar(models, [m1['mae'], m2['mae']], color=colors, alpha=0.8)
        axes[0, 2].set_ylabel('MAE (mm)')
        axes[0, 2].set_title('Mean Absolute Error', fontsize=12)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 4. Delta Accuracy
        deltas = ['δ1', 'δ2', 'δ3']
        d1_vals = [m1['delta_1'], m2['delta_1']]
        d2_vals = [m1['delta_2'], m2['delta_2']]
        d3_vals = [m1['delta_3'], m2['delta_3']]
        
        x = np.arange(len(deltas))
        width = 0.35
        axes[1, 0].bar(x - width/2, [d1_vals[0], d2_vals[0], d3_vals[0]], width, label=self.name1, color=colors[0], alpha=0.8)
        axes[1, 0].bar(x + width/2, [d1_vals[1], d2_vals[1], d3_vals[1]], width, label=self.name2, color=colors[1], alpha=0.8)
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Delta Accuracy (δ < 1.25^n)', fontsize=12)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(deltas)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1])
        
        # 5. Spearman Correlation
        axes[1, 1].bar(models, [m1['spearman'], m2['spearman']], color=colors, alpha=0.8)
        axes[1, 1].set_ylabel('Spearman ρ')
        axes[1, 1].set_title('Spearman Correlation', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim([0, 1])
        
        # 6. SILog
        axes[1, 2].bar(models, [m1['silog'], m2['silog']], color=colors, alpha=0.8)
        axes[1, 2].set_ylabel('SILog')
        axes[1, 2].set_title('Scale-Invariant Log Error', fontsize=12)
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ 전체 메트릭 비교: {self.output_dir / 'overall_metrics_comparison.png'}")
        plt.close()
    
    def _plot_distance_metrics(self):
        """거리별 메트릭 비교"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        dist_labels = [label for _, _, label in self.DISTANCE_RANGES]
        
        # 데이터 추출
        dist1 = self.results['distance_analysis'][self.name1]
        dist2 = self.results['distance_analysis'][self.name2]
        
        # 각 거리별로 메트릭 수집
        abs_rel_1 = [dist1.get(label, {}).get('abs_rel', 0) for label in dist_labels]
        abs_rel_2 = [dist2.get(label, {}).get('abs_rel', 0) for label in dist_labels]
        
        rmse_1 = [dist1.get(label, {}).get('rmse', 0) for label in dist_labels]
        rmse_2 = [dist2.get(label, {}).get('rmse', 0) for label in dist_labels]
        
        mae_1 = [dist1.get(label, {}).get('mae', 0) for label in dist_labels]
        mae_2 = [dist2.get(label, {}).get('mae', 0) for label in dist_labels]
        
        delta1_1 = [dist1.get(label, {}).get('delta_1', 0) for label in dist_labels]
        delta1_2 = [dist2.get(label, {}).get('delta_1', 0) for label in dist_labels]
        
        spearman_1 = [dist1.get(label, {}).get('spearman', 0) for label in dist_labels]
        spearman_2 = [dist2.get(label, {}).get('spearman', 0) for label in dist_labels]
        
        silog_1 = [dist1.get(label, {}).get('silog', 0) for label in dist_labels]
        silog_2 = [dist2.get(label, {}).get('silog', 0) for label in dist_labels]
        
        x = np.arange(len(dist_labels))
        width = 0.35
        
        # 1. AbsRel by distance
        axes[0, 0].bar(x - width/2, abs_rel_1, width, label=self.name1, color='steelblue', alpha=0.8)
        axes[0, 0].bar(x + width/2, abs_rel_2, width, label=self.name2, color='coral', alpha=0.8)
        axes[0, 0].set_ylabel('AbsRel')
        axes[0, 0].set_title('AbsRel by Distance', fontsize=12)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(dist_labels, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. RMSE by distance
        axes[0, 1].bar(x - width/2, rmse_1, width, label=self.name1, color='steelblue', alpha=0.8)
        axes[0, 1].bar(x + width/2, rmse_2, width, label=self.name2, color='coral', alpha=0.8)
        axes[0, 1].set_ylabel('RMSE (mm)')
        axes[0, 1].set_title('RMSE by Distance', fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(dist_labels, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. MAE by distance
        axes[0, 2].bar(x - width/2, mae_1, width, label=self.name1, color='steelblue', alpha=0.8)
        axes[0, 2].bar(x + width/2, mae_2, width, label=self.name2, color='coral', alpha=0.8)
        axes[0, 2].set_ylabel('MAE (mm)')
        axes[0, 2].set_title('MAE by Distance', fontsize=12)
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(dist_labels, rotation=45, ha='right')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 4. Delta-1 by distance
        axes[1, 0].bar(x - width/2, delta1_1, width, label=self.name1, color='steelblue', alpha=0.8)
        axes[1, 0].bar(x + width/2, delta1_2, width, label=self.name2, color='coral', alpha=0.8)
        axes[1, 0].set_ylabel('δ1 Accuracy')
        axes[1, 0].set_title('δ1 by Distance', fontsize=12)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(dist_labels, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1])
        
        # 5. Spearman by distance
        axes[1, 1].bar(x - width/2, spearman_1, width, label=self.name1, color='steelblue', alpha=0.8)
        axes[1, 1].bar(x + width/2, spearman_2, width, label=self.name2, color='coral', alpha=0.8)
        axes[1, 1].set_ylabel('Spearman ρ')
        axes[1, 1].set_title('Spearman by Distance', fontsize=12)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(dist_labels, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_ylim([0, 1])
        
        # 6. SILog by distance
        axes[1, 2].bar(x - width/2, silog_1, width, label=self.name1, color='steelblue', alpha=0.8)
        axes[1, 2].bar(x + width/2, silog_2, width, label=self.name2, color='coral', alpha=0.8)
        axes[1, 2].set_ylabel('SILog')
        axes[1, 2].set_title('SILog by Distance', fontsize=12)
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(dist_labels, rotation=45, ha='right')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distance_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ 거리별 메트릭 비교: {self.output_dir / 'distance_metrics_comparison.png'}")
        plt.close()
    
    def _plot_error_maps(self):
        """에러 맵 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        mask1 = self._create_valid_mask(self.pred1)
        mask2 = self._create_valid_mask(self.pred2)
        
        # Scale alignment for pred1
        scale1, shift1 = self._compute_scale_and_shift(self.pred1, self.gt, mask1)
        pred1_aligned = self.pred1 * scale1 + shift1
        
        # Absolute errors
        error1 = np.abs(pred1_aligned - self.gt)
        error2 = np.abs(self.pred2 - self.gt)
        
        error1_masked = np.ma.masked_where(~mask1, error1)
        error2_masked = np.ma.masked_where(~mask2, error2)
        
        # Relative errors
        rel_error1 = np.abs(pred1_aligned - self.gt) / self.gt
        rel_error2 = np.abs(self.pred2 - self.gt) / self.gt
        
        rel_error1_masked = np.ma.masked_where(~mask1, rel_error1)
        rel_error2_masked = np.ma.masked_where(~mask2, rel_error2)
        
        # Error visualization range
        error_vmax = np.percentile(error1[mask1], 95)
        rel_error_vmax = np.percentile(rel_error1[mask1], 95)
        
        # 1. Absolute Error - Pred1
        im0 = axes[0, 0].imshow(error1_masked, cmap='RdYlGn_r', vmin=0, vmax=error_vmax)
        axes[0, 0].set_title(f'{self.name1} - Absolute Error', fontsize=12)
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Error (mm)')
        
        # 2. Absolute Error - Pred2
        im1 = axes[0, 1].imshow(error2_masked, cmap='RdYlGn_r', vmin=0, vmax=error_vmax)
        axes[0, 1].set_title(f'{self.name2} - Absolute Error', fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label='Error (mm)')
        
        # 3. Relative Error - Pred1
        im2 = axes[1, 0].imshow(rel_error1_masked, cmap='RdYlGn_r', vmin=0, vmax=rel_error_vmax)
        axes[1, 0].set_title(f'{self.name1} - Relative Error', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Relative Error')
        
        # 4. Relative Error - Pred2
        im3 = axes[1, 1].imshow(rel_error2_masked, cmap='RdYlGn_r', vmin=0, vmax=rel_error_vmax)
        axes[1, 1].set_title(f'{self.name2} - Relative Error', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Relative Error')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_maps_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ 에러 맵 비교: {self.output_dir / 'error_maps_comparison.png'}")
        plt.close()
    
    def generate_report(self):
        """비교 리포트 생성"""
        print("4. 리포트 생성 중...")
        
        # 1. JSON 저장
        self._save_json()
        
        # 2. CSV 저장
        self._save_csv()
        
        # 3. Markdown 리포트
        self._save_markdown()
        
        print("✓ 리포트 생성 완료\n")
    
    def _save_json(self):
        """JSON 형식으로 저장"""
        output_path = self.output_dir / 'comparison_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"  ✓ JSON 리포트: {output_path}")
    
    def _save_csv(self):
        """CSV 형식으로 저장"""
        # 전체 메트릭 CSV
        m1 = self.results['overall'][self.name1]['with_alignment']
        m2 = self.results['overall'][self.name2]['without_alignment']
        
        csv_path = self.output_dir / 'overall_comparison.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'AbsRel', 'RMSE (mm)', 'MAE (mm)', 'δ1', 'δ2', 'δ3', 'Spearman', 'SILog', 'Num_Pixels'])
            
            for name, metrics in [(self.name1, m1), (self.name2, m2)]:
                writer.writerow([
                    name,
                    f"{metrics['abs_rel']:.4f}",
                    f"{metrics['rmse']:.2f}",
                    f"{metrics['mae']:.2f}",
                    f"{metrics['delta_1']:.4f}",
                    f"{metrics['delta_2']:.4f}",
                    f"{metrics['delta_3']:.4f}",
                    f"{metrics['spearman']:.4f}",
                    f"{metrics['silog']:.4f}",
                    metrics['num_pixels']
                ])
        print(f"  ✓ 전체 메트릭 CSV: {csv_path}")
        
        # 거리별 메트릭 CSV
        csv_path = self.output_dir / 'distance_comparison.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Distance', 'Model', 'abs_rel', 'rmse', 'mae', 'delta_1', 'delta_2', 'delta_3', 'spearman', 'silog', 'num_pixels'])
            
            for label in [l for _, _, l in self.DISTANCE_RANGES]:
                m1_dist = self.results['distance_analysis'][self.name1].get(label, {})
                m2_dist = self.results['distance_analysis'][self.name2].get(label, {})
                
                if m1_dist:
                    writer.writerow([
                        label, self.name1,
                        f"{m1_dist.get('abs_rel', 0):.4f}",
                        f"{m1_dist.get('rmse', 0):.2f}",
                        f"{m1_dist.get('mae', 0):.2f}",
                        f"{m1_dist.get('delta_1', 0):.4f}",
                        f"{m1_dist.get('delta_2', 0):.4f}",
                        f"{m1_dist.get('delta_3', 0):.4f}",
                        f"{m1_dist.get('spearman', 0):.4f}",
                        f"{m1_dist.get('silog', 0):.4f}",
                        m1_dist.get('num_pixels', 0)
                    ])
                
                if m2_dist:
                    writer.writerow([
                        label, self.name2,
                        f"{m2_dist.get('abs_rel', 0):.4f}",
                        f"{m2_dist.get('rmse', 0):.2f}",
                        f"{m2_dist.get('mae', 0):.2f}",
                        f"{m2_dist.get('delta_1', 0):.4f}",
                        f"{m2_dist.get('delta_2', 0):.4f}",
                        f"{m2_dist.get('delta_3', 0):.4f}",
                        f"{m2_dist.get('spearman', 0):.4f}",
                        f"{m2_dist.get('silog', 0):.4f}",
                        m2_dist.get('num_pixels', 0)
                    ])
        print(f"  ✓ 거리별 메트릭 CSV: {csv_path}")
    
    def _save_markdown(self):
        """Markdown 리포트 생성"""
        from datetime import datetime
        
        report = []
        report.append(f"# Depth Map 비교 리포트\n\n")
        report.append(f"**모델 1**: {self.name1}\n")
        report.append(f"**모델 2**: {self.name2}\n")
        report.append(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. 전체 메트릭 비교
        report.append("## 1. 전체 메트릭 비교\n\n")
        
        m1 = self.results['overall'][self.name1]['with_alignment']
        m2 = self.results['overall'][self.name2]['without_alignment']
        
        report.append("| 메트릭 | " + self.name1 + " | " + self.name2 + " | 차이 | 개선율 |\n")
        report.append("|--------|---------|---------|------|--------|\n")
        
        # AbsRel
        diff_absrel = m2['abs_rel'] - m1['abs_rel']
        improve_absrel = (diff_absrel / m1['abs_rel']) * 100 if m1['abs_rel'] != 0 else 0
        report.append(f"| AbsRel | {m1['abs_rel']:.4f} | {m2['abs_rel']:.4f} | {diff_absrel:+.4f} | {-improve_absrel:+.1f}% |\n")
        
        # RMSE
        diff_rmse = m2['rmse'] - m1['rmse']
        improve_rmse = (diff_rmse / m1['rmse']) * 100 if m1['rmse'] != 0 else 0
        report.append(f"| RMSE (mm) | {m1['rmse']:.2f} | {m2['rmse']:.2f} | {diff_rmse:+.2f} | {-improve_rmse:+.1f}% |\n")
        
        # MAE
        diff_mae = m2['mae'] - m1['mae']
        improve_mae = (diff_mae / m1['mae']) * 100 if m1['mae'] != 0 else 0
        report.append(f"| MAE (mm) | {m1['mae']:.2f} | {m2['mae']:.2f} | {diff_mae:+.2f} | {-improve_mae:+.1f}% |\n")
        
        # Delta 1/2/3
        for i, delta in enumerate(['δ1', 'δ2', 'δ3'], 1):
            key = f'delta_{i}'
            diff = m2[key] - m1[key]
            improve = (diff / m1[key]) * 100 if m1[key] != 0 else 0
            report.append(f"| {delta} | {m1[key]:.4f} | {m2[key]:.4f} | {diff:+.4f} | {improve:+.1f}% |\n")
        
        # Spearman
        diff_spear = m2['spearman'] - m1['spearman']
        improve_spear = (diff_spear / m1['spearman']) * 100 if m1['spearman'] != 0 else 0
        report.append(f"| Spearman | {m1['spearman']:.4f} | {m2['spearman']:.4f} | {diff_spear:+.4f} | {improve_spear:+.1f}% |\n")
        
        # SILog
        diff_silog = m2['silog'] - m1['silog']
        improve_silog = (diff_silog / m1['silog']) * 100 if m1['silog'] != 0 else 0
        report.append(f"| SILog | {m1['silog']:.4f} | {m2['silog']:.4f} | {diff_silog:+.4f} | {-improve_silog:+.1f}% |\n")
        
        report.append("\n")
        
        # 2. 거리별 분석
        report.append("## 2. 거리별 성능 비교\n\n")
        report.append("### AbsRel by Distance\n\n")
        report.append("| 거리 범위 | " + self.name1 + " | " + self.name2 + " | 차이 |\n")
        report.append("|-----------|---------|---------|------|\n")
        
        for label in [l for _, _, l in self.DISTANCE_RANGES]:
            m1_dist = self.results['distance_analysis'][self.name1].get(label, {})
            m2_dist = self.results['distance_analysis'][self.name2].get(label, {})
            
            if m1_dist and m2_dist:
                diff = m2_dist['abs_rel'] - m1_dist['abs_rel']
                report.append(f"| {label} | {m1_dist['abs_rel']:.4f} | {m2_dist['abs_rel']:.4f} | {diff:+.4f} |\n")
        
        report.append("\n### RMSE by Distance\n\n")
        report.append("| 거리 범위 | " + self.name1 + " | " + self.name2 + " | 차이 |\n")
        report.append("|-----------|---------|---------|------|\n")
        
        for label in [l for _, _, l in self.DISTANCE_RANGES]:
            m1_dist = self.results['distance_analysis'][self.name1].get(label, {})
            m2_dist = self.results['distance_analysis'][self.name2].get(label, {})
            
            if m1_dist and m2_dist:
                diff = m2_dist['rmse'] - m1_dist['rmse']
                report.append(f"| {label} | {m1_dist['rmse']:.2f} | {m2_dist['rmse']:.2f} | {diff:+.2f} |\n")
        
        report.append("\n## 3. 요약\n\n")
        
        # 승자 판정
        wins = {self.name1: 0, self.name2: 0}
        
        # Error metrics (낮을수록 좋음)
        if m1['abs_rel'] < m2['abs_rel']:
            wins[self.name1] += 1
        else:
            wins[self.name2] += 1
        
        if m1['rmse'] < m2['rmse']:
            wins[self.name1] += 1
        else:
            wins[self.name2] += 1
        
        if m1['mae'] < m2['mae']:
            wins[self.name1] += 1
        else:
            wins[self.name2] += 1
        
        # Accuracy metrics (높을수록 좋음)
        for i in range(1, 4):
            if m1[f'delta_{i}'] > m2[f'delta_{i}']:
                wins[self.name1] += 1
            else:
                wins[self.name2] += 1
        
        if m1['spearman'] > m2['spearman']:
            wins[self.name1] += 1
        else:
            wins[self.name2] += 1
        
        # SILog (낮을수록 좋음)
        if m1['silog'] < m2['silog']:
            wins[self.name1] += 1
        else:
            wins[self.name2] += 1
        
        report.append(f"- **{self.name1}**: {wins[self.name1]}/8 메트릭에서 우수\n")
        report.append(f"- **{self.name2}**: {wins[self.name2]}/8 메트릭에서 우수\n\n")
        
        if wins[self.name1] > wins[self.name2]:
            report.append(f"**결론**: {self.name1}이(가) 전반적으로 더 나은 성능을 보입니다.\n")
        elif wins[self.name2] > wins[self.name1]:
            report.append(f"**결론**: {self.name2}이(가) 전반적으로 더 나은 성능을 보입니다.\n")
        else:
            report.append(f"**결론**: 두 모델이 비슷한 성능을 보입니다.\n")
        
        # 저장
        md_path = self.output_dir / 'comparison_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(''.join(report))
        print(f"  ✓ Markdown 리포트: {md_path}")
    
    def run(self):
        """전체 비교 프로세스 실행"""
        self.compare()
        self.visualize()
        self.generate_report()
        
        print("=" * 60)
        print("비교 완료!")
        print(f"결과 저장 위치: {self.output_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="두 Depth Map (numpy 파일) 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python compare_depth_images.py \\
      --pred1 depth_output_rel/origin/depth_npy/000001.npy \\
      --pred2 depth_output_abs/origin/depth_npy/000001.npy \\
      --gt depth_output_zed/origin/depth_npy/000001.npy \\
      --name1 "Monocular" \\
      --name2 "Metric" \\
      --output comparison_results
        """
    )
    
    parser.add_argument('--pred1', type=str, required=True,
                        help='첫 번째 예측 depth map (.npy)')
    parser.add_argument('--pred2', type=str, required=True,
                        help='두 번째 예측 depth map (.npy)')
    parser.add_argument('--gt', type=str, required=True,
                        help='Ground truth depth map (.npy)')
    parser.add_argument('--name1', type=str, default='Pred1',
                        help='첫 번째 모델 이름 (기본값: Pred1)')
    parser.add_argument('--name2', type=str, default='Pred2',
                        help='두 번째 모델 이름 (기본값: Pred2)')
    parser.add_argument('--confidence', type=str, default=None,
                        help='Confidence map (.npy, 선택사항)')
    parser.add_argument('--confidence_threshold', type=float, default=30.0,
                        help='Confidence threshold (0-100, 기본값: 30)')
    parser.add_argument('--max_distance', type=float, default=20000.0,
                        help='최대 거리 (mm, 기본값: 20000)')
    parser.add_argument('--min_distance', type=float, default=200.0,
                        help='최소 거리 (mm, 기본값: 200)')
    parser.add_argument('--output', type=str, default='comparison_results',
                        help='결과 저장 디렉토리 (기본값: comparison_results)')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    for path, name in [(args.pred1, 'pred1'), (args.pred2, 'pred2'), (args.gt, 'gt')]:
        if not Path(path).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {path}")
            return
    
    # 비교 실행
    comparator = DepthComparator(
        pred1_path=args.pred1,
        pred2_path=args.pred2,
        gt_path=args.gt,
        name1=args.name1,
        name2=args.name2,
        confidence_path=args.confidence,
        confidence_threshold=args.confidence_threshold,
        max_distance=args.max_distance,
        min_distance=args.min_distance,
        output_dir=args.output
    )
    
    comparator.run()


if __name__ == "__main__":
    main()

