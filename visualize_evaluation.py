"""
평가 결과 시각화 및 리포트 생성 (개선 버전)
- 거리별 모든 메트릭 CSV 저장
- Monocular / Metric 모델 비교
- 피벗 테이블 형태 CSV 지원
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings


class EvaluationVisualizer:
    """평가 결과 시각화 클래스"""
    
    # 클래스 상수
    DISTANCE_RANGES = ['0.0-1.0m', '1.0-2.0m', '2.0-5.0m', '5.0-10.0m', '10.0-20.0m']
    CONFIDENCE_RANGES = ['0-20', '20-40', '40-60', '60-80', '80-100']
    METRICS = ['abs_rel', 'rmse', 'mae', 'delta_1', 'delta_2', 'delta_3', 'silog']
    
    def __init__(self, results_path: str):
        """
        Args:
            results_path: evaluation_results.json 파일 경로
        """
        self.results_path = Path(results_path)
        with open(self.results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # 출력 디렉토리
        self.output_dir = self.results_path.parent
        
        # 데이터 확인
        self._validate_results()
        
        # 스타일 설정
        self._setup_plot_style()
    
    def _validate_results(self):
        """결과 데이터 유효성 검증"""
        if not self.results:
            print("Warning: Results file is empty!")
        elif 'relative' not in self.results and 'absolute' not in self.results:
            print("Warning: No evaluation results found in the file!")
        else:
            if 'relative' in self.results:
                print("✓ Monocular (relative) 모델 결과 발견")
            if 'absolute' in self.results:
                print("✓ Metric (absolute) 모델 결과 발견")
    
    def _setup_plot_style(self):
        """플롯 스타일 설정"""
        # matplotlib 스타일
        for style in ['seaborn-v0_8-darkgrid', 'seaborn-darkgrid', 'default']:
            try:
                plt.style.use(style)
                break
            except OSError:
                continue
        
        sns.set_palette("husl")
        
        # 폰트 설정
        import matplotlib
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    def _get_nested_value(self, d: Dict, key_path: str, default: Optional[float] = None) -> Optional[float]:
        """
        중첩된 딕셔너리에서 값 가져오기
        
        Args:
            d: 딕셔너리
            key_path: 점으로 구분된 키 경로 (예: "after_alignment.abs_rel.mean")
            default: 기본값 (None 권장 - 명시적 처리를 위해)
        
        Returns:
            찾은 값 또는 default
        """
        if d is None:
            return default
            
        keys = key_path.split('.')
        
        # 마지막 키가 'mean', 'std' 등인지 확인
        if len(keys) >= 2 and keys[-1] in ['mean', 'std', 'min', 'max']:
            main_key = '.'.join(keys[:-1])
            sub_key = keys[-1]
            
            if main_key in d:
                val = d[main_key]
                if isinstance(val, dict) and sub_key in val:
                    return float(val[sub_key])
        
        # 기존 방식 (완전히 중첩된 경우)
        val = d
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        
        if isinstance(val, dict) and 'mean' in val:
            return float(val['mean'])
        
        if isinstance(val, (int, float)):
            return float(val)
        
        return default
    
    def _get_distance_metrics(self, model_key: str) -> List[Dict[str, Any]]:
        """
        특정 모델의 거리별 메트릭 추출
        
        Args:
            model_key: 'relative' 또는 'absolute'
        
        Returns:
            거리별 메트릭 리스트
        """
        if model_key not in self.results:
            return []
        
        result = self.results[model_key]
        data = []
        
        for dist_range in self.DISTANCE_RANGES:
            row = {'distance_range': dist_range}
            has_data = False
            
            for metric in self.METRICS:
                mean_val = self._get_nested_value(
                    result, f'distance_analysis.{dist_range}.{metric}.mean'
                )
                std_val = self._get_nested_value(
                    result, f'distance_analysis.{dist_range}.{metric}.std'
                )
                
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                
                if mean_val is not None:
                    has_data = True
            
            if has_data:
                data.append(row)
        
        return data
    
    def plot_metric_comparison(self):
        """Delta Accuracy 비교 차트"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Delta accuracy 비교
        if 'relative' in self.results and 'absolute' in self.results:
            deltas = ['δ1', 'δ2', 'δ3']
            rel_deltas = [
                self._get_nested_value(self.results['relative'], 'after_alignment.delta_1.mean', 0),
                self._get_nested_value(self.results['relative'], 'after_alignment.delta_2.mean', 0),
                self._get_nested_value(self.results['relative'], 'after_alignment.delta_3.mean', 0)
            ]
            abs_deltas = [
                self._get_nested_value(self.results['absolute'], 'direct_comparison.delta_1.mean', 0),
                self._get_nested_value(self.results['absolute'], 'direct_comparison.delta_2.mean', 0),
                self._get_nested_value(self.results['absolute'], 'direct_comparison.delta_3.mean', 0)
            ]
            
            width = 0.35
            x_delta = np.arange(len(deltas))
            ax.bar(x_delta - width/2, rel_deltas, width, label='Monocular (DA3-Mono)', alpha=0.8, color='steelblue')
            ax.bar(x_delta + width/2, abs_deltas, width, label='Metric (DA3-Metric)', alpha=0.8, color='coral')
            ax.set_ylabel('Accuracy')
            ax.set_title('Delta Accuracy Comparison', fontsize=12)
            ax.set_xticks(x_delta)
            ax.set_xticklabels(deltas)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'delta_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Delta Accuracy 비교 차트 저장: {self.output_dir / 'delta_accuracy_comparison.png'}")
        plt.close()
    
    def plot_distance_analysis(self):
        """거리별 성능 분석 차트"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        x_dist = np.arange(len(self.DISTANCE_RANGES))
        width = 0.35
        
        # 데이터 추출
        rel_data = {metric: [] for metric in self.METRICS}
        abs_data = {metric: [] for metric in self.METRICS}
        
        for dist_range in self.DISTANCE_RANGES:
            if 'relative' in self.results:
                rel = self.results['relative']
                for metric in self.METRICS:
                    val = self._get_nested_value(rel, f'distance_analysis.{dist_range}.{metric}.mean', 0)
                    rel_data[metric].append(val if val is not None else 0)
            
            if 'absolute' in self.results:
                abs_ = self.results['absolute']
                for metric in self.METRICS:
                    val = self._get_nested_value(abs_, f'distance_analysis.{dist_range}.{metric}.mean', 0)
                    abs_data[metric].append(val if val is not None else 0)
        
        # 1. RMSE 비교
        if rel_data['rmse'] and abs_data['rmse']:
            axes[0, 0].bar(x_dist - width/2, rel_data['rmse'], width, alpha=0.7, color='steelblue', label='Monocular')
            axes[0, 0].bar(x_dist + width/2, abs_data['rmse'], width, alpha=0.7, color='coral', label='Metric')
            axes[0, 0].set_xlabel('Distance Range')
            axes[0, 0].set_ylabel('RMSE (mm)')
            axes[0, 0].set_title('RMSE by Distance', fontsize=12)
            axes[0, 0].set_xticks(x_dist)
            axes[0, 0].set_xticklabels(self.DISTANCE_RANGES, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. AbsRel 비교
        if rel_data['abs_rel'] and abs_data['abs_rel']:
            axes[0, 1].bar(x_dist - width/2, rel_data['abs_rel'], width, alpha=0.7, color='steelblue', label='Monocular')
            axes[0, 1].bar(x_dist + width/2, abs_data['abs_rel'], width, alpha=0.7, color='coral', label='Metric')
            axes[0, 1].set_xlabel('Distance Range')
            axes[0, 1].set_ylabel('AbsRel')
            axes[0, 1].set_title('AbsRel by Distance', fontsize=12)
            axes[0, 1].set_xticks(x_dist)
            axes[0, 1].set_xticklabels(self.DISTANCE_RANGES, rotation=45, ha='right')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Delta-1 비교
        if rel_data['delta_1'] and abs_data['delta_1']:
            axes[0, 2].bar(x_dist - width/2, rel_data['delta_1'], width, alpha=0.7, color='steelblue', label='Monocular')
            axes[0, 2].bar(x_dist + width/2, abs_data['delta_1'], width, alpha=0.7, color='coral', label='Metric')
            axes[0, 2].set_xlabel('Distance Range')
            axes[0, 2].set_ylabel('δ1 Accuracy')
            axes[0, 2].set_title('δ1 by Distance', fontsize=12)
            axes[0, 2].set_xticks(x_dist)
            axes[0, 2].set_xticklabels(self.DISTANCE_RANGES, rotation=45, ha='right')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3, axis='y')
            axes[0, 2].set_ylim([0, 1])
        
        # 4. Monocular 모델 RMSE & MAE
        if 'relative' in self.results:
            axes[1, 0].bar(x_dist - width/2, rel_data['rmse'], width, alpha=0.7, color='steelblue', label='RMSE')
            axes[1, 0].bar(x_dist + width/2, rel_data['mae'], width, alpha=0.7, color='lightblue', label='MAE')
            axes[1, 0].set_xlabel('Distance Range')
            axes[1, 0].set_ylabel('Error (mm)')
            axes[1, 0].set_title('Monocular: RMSE & MAE by Distance', fontsize=12)
            axes[1, 0].set_xticks(x_dist)
            axes[1, 0].set_xticklabels(self.DISTANCE_RANGES, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. Metric 모델 RMSE & MAE
        if 'absolute' in self.results:
            axes[1, 1].bar(x_dist - width/2, abs_data['rmse'], width, alpha=0.7, color='coral', label='RMSE')
            axes[1, 1].bar(x_dist + width/2, abs_data['mae'], width, alpha=0.7, color='lightsalmon', label='MAE')
            axes[1, 1].set_xlabel('Distance Range')
            axes[1, 1].set_ylabel('Error (mm)')
            axes[1, 1].set_title('Metric: RMSE & MAE by Distance', fontsize=12)
            axes[1, 1].set_xticks(x_dist)
            axes[1, 1].set_xticklabels(self.DISTANCE_RANGES, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. SILog 비교
        rel_silog = rel_data.get('silog', [])
        abs_silog = abs_data.get('silog', [])
        
        # 유효한 값이 있는지 확인
        rel_silog_valid = [v for v in rel_silog if v and v != 0]
        abs_silog_valid = [v for v in abs_silog if v and v != 0]
        
        if rel_silog_valid or abs_silog_valid:
            axes[1, 2].bar(x_dist - width/2, rel_silog, width, alpha=0.7, color='steelblue', label='Monocular')
            axes[1, 2].bar(x_dist + width/2, abs_silog, width, alpha=0.7, color='coral', label='Metric')
            axes[1, 2].set_xlabel('Distance Range')
            axes[1, 2].set_ylabel('SILog')
            axes[1, 2].set_title('SILog by Distance', fontsize=12)
            axes[1, 2].set_xticks(x_dist)
            axes[1, 2].set_xticklabels(self.DISTANCE_RANGES, rotation=45, ha='right')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3, axis='y')
        else:
            # 거리별 SILog가 없으면 전체 값 표시
            models = []
            silog_values = []
            colors = []
            
            if 'relative' in self.results:
                val = self._get_nested_value(self.results['relative'], 'scale_invariant.silog.mean')
                if val is not None:
                    models.append('Monocular')
                    silog_values.append(val)
                    colors.append('steelblue')
            
            if 'absolute' in self.results:
                val = self._get_nested_value(self.results['absolute'], 'scale_invariant.silog.mean')
                if val is not None:
                    models.append('Metric')
                    silog_values.append(val)
                    colors.append('coral')
            
            if models:
                axes[1, 2].bar(np.arange(len(models)), silog_values, alpha=0.7, color=colors)
                axes[1, 2].set_ylabel('SILog')
                axes[1, 2].set_title('SILog (Global)', fontsize=12)
                axes[1, 2].set_xticks(np.arange(len(models)))
                axes[1, 2].set_xticklabels(models)
                axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ 거리별 분석 차트 저장: {self.output_dir / 'distance_analysis.png'}")
        plt.close()
    
    def plot_confidence_analysis(self):
        """Confidence별 성능 분석 차트"""
        # Confidence 데이터가 있는지 확인
        has_conf_data = False
        if 'relative' in self.results:
            for conf_range in self.CONFIDENCE_RANGES:
                val = self._get_nested_value(
                    self.results['relative'], f'confidence_analysis.{conf_range}.rmse.mean'
                )
                if val is not None:
                    has_conf_data = True
                    break
        
        if not has_conf_data:
            print("  ⚠ Confidence 분석 데이터가 없습니다. 차트 생성을 건너뜁니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x_conf = np.arange(len(self.CONFIDENCE_RANGES))
        width = 0.35
        
        # 데이터 추출
        rel_data = {metric: [] for metric in ['rmse', 'abs_rel', 'delta_1', 'pixel_ratio']}
        abs_data = {metric: [] for metric in ['rmse', 'abs_rel', 'delta_1', 'pixel_ratio']}
        
        for conf_range in self.CONFIDENCE_RANGES:
            if 'relative' in self.results:
                rel = self.results['relative']
                for metric in ['rmse', 'abs_rel', 'delta_1', 'pixel_ratio']:
                    val = self._get_nested_value(rel, f'confidence_analysis.{conf_range}.{metric}.mean', 0)
                    rel_data[metric].append(val if val is not None else 0)
            
            if 'absolute' in self.results:
                abs_ = self.results['absolute']
                for metric in ['rmse', 'abs_rel', 'delta_1', 'pixel_ratio']:
                    val = self._get_nested_value(abs_, f'confidence_analysis.{conf_range}.{metric}.mean', 0)
                    abs_data[metric].append(val if val is not None else 0)
        
        # 1. Confidence별 RMSE 비교
        if rel_data['rmse'] and abs_data['rmse']:
            axes[0, 0].bar(x_conf - width/2, rel_data['rmse'], width, alpha=0.7, color='steelblue', label='Monocular')
            axes[0, 0].bar(x_conf + width/2, abs_data['rmse'], width, alpha=0.7, color='coral', label='Metric')
            axes[0, 0].set_xlabel('Confidence Range')
            axes[0, 0].set_ylabel('RMSE (mm)')
            axes[0, 0].set_title('RMSE by Confidence', fontsize=12)
            axes[0, 0].set_xticks(x_conf)
            axes[0, 0].set_xticklabels(self.CONFIDENCE_RANGES, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Confidence별 AbsRel 비교
        if rel_data['abs_rel'] and abs_data['abs_rel']:
            axes[0, 1].bar(x_conf - width/2, rel_data['abs_rel'], width, alpha=0.7, color='steelblue', label='Monocular')
            axes[0, 1].bar(x_conf + width/2, abs_data['abs_rel'], width, alpha=0.7, color='coral', label='Metric')
            axes[0, 1].set_xlabel('Confidence Range')
            axes[0, 1].set_ylabel('AbsRel')
            axes[0, 1].set_title('AbsRel by Confidence', fontsize=12)
            axes[0, 1].set_xticks(x_conf)
            axes[0, 1].set_xticklabels(self.CONFIDENCE_RANGES, rotation=45, ha='right')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Confidence별 Delta-1 비교
        if rel_data['delta_1'] and abs_data['delta_1']:
            axes[1, 0].bar(x_conf - width/2, rel_data['delta_1'], width, alpha=0.7, color='steelblue', label='Monocular')
            axes[1, 0].bar(x_conf + width/2, abs_data['delta_1'], width, alpha=0.7, color='coral', label='Metric')
            axes[1, 0].set_xlabel('Confidence Range')
            axes[1, 0].set_ylabel('δ1 Accuracy')
            axes[1, 0].set_title('δ1 by Confidence', fontsize=12)
            axes[1, 0].set_xticks(x_conf)
            axes[1, 0].set_xticklabels(self.CONFIDENCE_RANGES, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].set_ylim([0, 1])
        
        # 4. Confidence별 픽셀 분포 (pixel_ratio)
        if rel_data['pixel_ratio']:
            # 비율을 퍼센트로 변환
            pixel_ratios = [r * 100 if r else 0 for r in rel_data['pixel_ratio']]
            axes[1, 1].bar(x_conf, pixel_ratios, alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Confidence Range')
            axes[1, 1].set_ylabel('Pixel Ratio (%)')
            axes[1, 1].set_title('Pixel Distribution by Confidence', fontsize=12)
            axes[1, 1].set_xticks(x_conf)
            axes[1, 1].set_xticklabels(self.CONFIDENCE_RANGES, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Confidence 분석 차트 저장: {self.output_dir / 'confidence_analysis.png'}")
        plt.close()
    
    def generate_summary_table(self):
        """요약 테이블 생성"""
        data = []
        
        if 'relative' in self.results:
            rel = self.results['relative']
            data.append({
                'Model': 'Monocular (DA3-Mono)',
                'AbsRel': f"{self._get_nested_value(rel, 'after_alignment.abs_rel.mean', 0):.4f}",
                'RMSE (mm)': f"{self._get_nested_value(rel, 'after_alignment.rmse.mean', 0):.2f}",
                'MAE (mm)': f"{self._get_nested_value(rel, 'after_alignment.mae.mean', 0):.2f}",
                'δ1': f"{self._get_nested_value(rel, 'after_alignment.delta_1.mean', 0):.4f}",
                'δ2': f"{self._get_nested_value(rel, 'after_alignment.delta_2.mean', 0):.4f}",
                'δ3': f"{self._get_nested_value(rel, 'after_alignment.delta_3.mean', 0):.4f}",
                'Spearman': f"{self._get_nested_value(rel, 'scale_invariant.spearman.mean', 0):.4f}",
                'SILog': f"{self._get_nested_value(rel, 'scale_invariant.silog.mean', 0):.4f}"
            })
        
        if 'absolute' in self.results:
            abs_ = self.results['absolute']
            spearman = self._get_nested_value(abs_, 'scale_invariant.spearman.mean')
            silog = self._get_nested_value(abs_, 'scale_invariant.silog.mean')
            
            data.append({
                'Model': 'Metric (DA3-Metric)',
                'AbsRel': f"{self._get_nested_value(abs_, 'direct_comparison.abs_rel.mean', 0):.4f}",
                'RMSE (mm)': f"{self._get_nested_value(abs_, 'direct_comparison.rmse.mean', 0):.2f}",
                'MAE (mm)': f"{self._get_nested_value(abs_, 'direct_comparison.mae.mean', 0):.2f}",
                'δ1': f"{self._get_nested_value(abs_, 'direct_comparison.delta_1.mean', 0):.4f}",
                'δ2': f"{self._get_nested_value(abs_, 'direct_comparison.delta_2.mean', 0):.4f}",
                'δ3': f"{self._get_nested_value(abs_, 'direct_comparison.delta_3.mean', 0):.4f}",
                'Spearman': f"{spearman:.4f}" if spearman is not None else '-',
                'SILog': f"{silog:.4f}" if silog is not None else '-'
            })
        
        if not data:
            print("⚠ 경고: 표시할 데이터가 없습니다.")
            return
        
        df = pd.DataFrame(data)
        
        # CSV로 저장
        df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        print(f"✓ 요약 테이블 저장: {self.output_dir / 'summary_table.csv'}")
        print("\n" + df.to_string(index=False))
    
    def save_overall_metrics_csv(self):
        """전체 메트릭 요약 CSV (거리 구분 없이)"""
        data = []
        
        metrics_map = {
            'relative': {
                'prefix': 'after_alignment',
                'model_name': 'Monocular (DA3-Mono)'
            },
            'absolute': {
                'prefix': 'direct_comparison',
                'model_name': 'Metric (DA3-Metric)'
            }
        }
        
        for key, info in metrics_map.items():
            if key not in self.results:
                continue
            
            result = self.results[key]
            prefix = info['prefix']
            
            row = {
                'model': info['model_name'],
                'abs_rel_mean': self._get_nested_value(result, f'{prefix}.abs_rel.mean'),
                'abs_rel_std': self._get_nested_value(result, f'{prefix}.abs_rel.std'),
                'rmse_mean': self._get_nested_value(result, f'{prefix}.rmse.mean'),
                'rmse_std': self._get_nested_value(result, f'{prefix}.rmse.std'),
                'mae_mean': self._get_nested_value(result, f'{prefix}.mae.mean'),
                'mae_std': self._get_nested_value(result, f'{prefix}.mae.std'),
                'delta_1_mean': self._get_nested_value(result, f'{prefix}.delta_1.mean'),
                'delta_1_std': self._get_nested_value(result, f'{prefix}.delta_1.std'),
                'delta_2_mean': self._get_nested_value(result, f'{prefix}.delta_2.mean'),
                'delta_2_std': self._get_nested_value(result, f'{prefix}.delta_2.std'),
                'delta_3_mean': self._get_nested_value(result, f'{prefix}.delta_3.mean'),
                'delta_3_std': self._get_nested_value(result, f'{prefix}.delta_3.std'),
                'silog_mean': self._get_nested_value(result, 'scale_invariant.silog.mean'),
                'silog_std': self._get_nested_value(result, 'scale_invariant.silog.std'),
                'spearman_mean': self._get_nested_value(result, 'scale_invariant.spearman.mean'),
                'spearman_std': self._get_nested_value(result, 'scale_invariant.spearman.std'),
            }
            data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            csv_path = self.output_dir / 'overall_metrics.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ 전체 메트릭 요약 CSV 저장: {csv_path}")
    
    def save_distance_metrics_csv(self):
        """거리별 모든 메트릭을 CSV로 저장 (Monocular & Metric 모델 비교)"""
        data = []
        
        # Monocular (상대 모델)
        if 'relative' in self.results:
            rel_metrics = self._get_distance_metrics('relative')
            for row in rel_metrics:
                row['model'] = 'Monocular (DA3-Mono)'
                data.append(row)
        
        # Metric (절대 모델)
        if 'absolute' in self.results:
            abs_metrics = self._get_distance_metrics('absolute')
            for row in abs_metrics:
                row['model'] = 'Metric (DA3-Metric)'
                data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            
            # 컬럼 순서 정리 (valid_pixels, pixel_ratio 제외)
            column_order = ['distance_range', 'model']
            for metric in self.METRICS:
                column_order.extend([f'{metric}_mean', f'{metric}_std'])
            
            # 존재하는 컬럼만 선택
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            # Long format 저장
            csv_path = self.output_dir / 'distance_metrics_detailed.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ 거리별 상세 메트릭 CSV 저장: {csv_path}")
        else:
            print("⚠ 경고: 거리별 메트릭 데이터가 없습니다.")
    
    def generate_text_report(self):
        """텍스트 리포트 생성"""
        report = []
        report.append("# Depth 모델 평가 리포트\n")
        report.append(f"생성 시간: {pd.Timestamp.now()}\n\n")
        
        # 1. Monocular 모델
        report.append("## 1. Monocular 깊이 모델 (DA3-Mono)\n")
        if 'relative' in self.results:
            rel = self.results['relative']
            report.append("### 주요 메트릭 (Scale Alignment 후)\n")
            report.append(f"- **AbsRel**: {self._get_nested_value(rel, 'after_alignment.abs_rel.mean', 0):.4f}\n")
            report.append(f"- **RMSE**: {self._get_nested_value(rel, 'after_alignment.rmse.mean', 0):.2f} mm\n")
            report.append(f"- **MAE**: {self._get_nested_value(rel, 'after_alignment.mae.mean', 0):.2f} mm\n")
            report.append(f"- **δ1**: {self._get_nested_value(rel, 'after_alignment.delta_1.mean', 0):.4f}\n")
            report.append(f"- **δ2**: {self._get_nested_value(rel, 'after_alignment.delta_2.mean', 0):.4f}\n")
            report.append(f"- **δ3**: {self._get_nested_value(rel, 'after_alignment.delta_3.mean', 0):.4f}\n")
            report.append(f"- **Spearman**: {self._get_nested_value(rel, 'scale_invariant.spearman.mean', 0):.4f}\n")
            report.append(f"- **SILog**: {self._get_nested_value(rel, 'scale_invariant.silog.mean', 0):.4f}\n\n")
        else:
            report.append("데이터 없음\n\n")
        
        # 2. Metric 모델
        report.append("## 2. Metric 깊이 모델 (DA3-Metric)\n")
        if 'absolute' in self.results:
            abs_ = self.results['absolute']
            report.append("### 주요 메트릭 (Direct Comparison)\n")
            report.append(f"- **AbsRel**: {self._get_nested_value(abs_, 'direct_comparison.abs_rel.mean', 0):.4f}\n")
            report.append(f"- **RMSE**: {self._get_nested_value(abs_, 'direct_comparison.rmse.mean', 0):.2f} mm\n")
            report.append(f"- **MAE**: {self._get_nested_value(abs_, 'direct_comparison.mae.mean', 0):.2f} mm\n")
            report.append(f"- **δ1**: {self._get_nested_value(abs_, 'direct_comparison.delta_1.mean', 0):.4f}\n")
            report.append(f"- **δ2**: {self._get_nested_value(abs_, 'direct_comparison.delta_2.mean', 0):.4f}\n")
            report.append(f"- **δ3**: {self._get_nested_value(abs_, 'direct_comparison.delta_3.mean', 0):.4f}\n\n")
            
            spearman = self._get_nested_value(abs_, 'scale_invariant.spearman.mean')
            silog = self._get_nested_value(abs_, 'scale_invariant.silog.mean')
            if spearman is not None or silog is not None:
                report.append("### Scale-invariant 메트릭\n")
                if spearman is not None:
                    report.append(f"- **Spearman**: {spearman:.4f}\n")
                if silog is not None:
                    report.append(f"- **SILog**: {silog:.4f}\n")
                report.append("\n")
        else:
            report.append("데이터 없음\n\n")
        
        # 3. 거리별 분석 요약
        report.append("## 3. 거리별 성능 분석\n")
        
        # 거리별 픽셀 분포 정보 추가
        if 'relative' in self.results or 'absolute' in self.results:
            report.append("### 거리별 픽셀 분포 (전체 유효 픽셀 대비)\n")
            
            # Monocular 모델 또는 Metric 모델 중 하나의 분포 정보 사용 (둘 다 동일해야 함)
            model_key = 'relative' if 'relative' in self.results else 'absolute'
            result = self.results[model_key]
            
            # filtered_ratio 데이터가 있는지 확인
            has_filtered_ratio = False
            for dist_range in self.DISTANCE_RANGES:
                filtered_ratio = self._get_nested_value(
                    result, f'distance_analysis.{dist_range}.filtered_ratio.mean'
                )
                if filtered_ratio is not None and filtered_ratio > 0:
                    has_filtered_ratio = True
                    break
            
            if has_filtered_ratio:
                report.append("| 거리 범위 | 픽셀 비율 | Confidence로 제외된 비율 |\n")
                report.append("|-----------|----------|-------------------------|\n")
                
                for dist_range in self.DISTANCE_RANGES:
                    pixel_ratio = self._get_nested_value(
                        result, f'distance_analysis.{dist_range}.pixel_ratio.mean'
                    )
                    filtered_ratio = self._get_nested_value(
                        result, f'distance_analysis.{dist_range}.filtered_ratio.mean'
                    )
                    if pixel_ratio is not None:
                        if filtered_ratio is not None:
                            report.append(f"| {dist_range} | {pixel_ratio*100:.2f}% | {filtered_ratio*100:.2f}% |\n")
                        else:
                            report.append(f"| {dist_range} | {pixel_ratio*100:.2f}% | N/A |\n")
            else:
                report.append("| 거리 범위 | 픽셀 비율 |\n")
                report.append("|-----------|----------|\n")
                
                for dist_range in self.DISTANCE_RANGES:
                    pixel_ratio = self._get_nested_value(
                        result, f'distance_analysis.{dist_range}.pixel_ratio.mean'
                    )
                    if pixel_ratio is not None:
                        report.append(f"| {dist_range} | {pixel_ratio*100:.2f}% |\n")
            
            report.append("\n")
        
        report.append("상세 데이터는 `distance_metrics_detailed.csv` 참조\n\n")
        
        # 4. Confidence별 분석 요약
        report.append("## 4. Confidence별 성능 분석\n")
        
        # Confidence 데이터가 있는지 확인
        has_conf_data = False
        if 'relative' in self.results or 'absolute' in self.results:
            model_key = 'relative' if 'relative' in self.results else 'absolute'
            result = self.results[model_key]
            
            for conf_range in self.CONFIDENCE_RANGES:
                if self._get_nested_value(result, f'confidence_analysis.{conf_range}.pixel_ratio.mean') is not None:
                    has_conf_data = True
                    break
        
        if has_conf_data:
            # Confidence별 픽셀 분포 정보
            report.append("### Confidence별 픽셀 분포 (전체 유효 픽셀 대비)\n")
            model_key = 'relative' if 'relative' in self.results else 'absolute'
            result = self.results[model_key]
            
            report.append("| Confidence 범위 | 픽셀 비율 |\n")
            report.append("|----------------|----------|\n")
            
            for conf_range in self.CONFIDENCE_RANGES:
                pixel_ratio = self._get_nested_value(
                    result, f'confidence_analysis.{conf_range}.pixel_ratio.mean'
                )
                if pixel_ratio is not None:
                    report.append(f"| {conf_range} | {pixel_ratio*100:.2f}% |\n")
            
            report.append("\n")
            
            # Confidence별 성능 요약
            report.append("### Confidence가 높을수록 성능이 좋은가?\n")
            report.append("Confidence 분석 차트(`confidence_analysis.png`)를 참조하세요.\n\n")
        else:
            report.append("Confidence map이 없어 분석을 수행하지 않았습니다.\n\n")
        
        # 5. 모델 비교
        report.append("## 5. 모델 비교 요약\n")
        if 'relative' in self.results and 'absolute' in self.results:
            rel = self.results['relative']
            abs_ = self.results['absolute']
            
            rel_absrel = self._get_nested_value(rel, 'after_alignment.abs_rel.mean', 0)
            abs_absrel = self._get_nested_value(abs_, 'direct_comparison.abs_rel.mean', 0)
            rel_rmse = self._get_nested_value(rel, 'after_alignment.rmse.mean', 0)
            abs_rmse = self._get_nested_value(abs_, 'direct_comparison.rmse.mean', 0)
            
            report.append("| 메트릭 | Monocular | Metric | 차이 |\n")
            report.append("|--------|-----------|--------|------|\n")
            report.append(f"| AbsRel | {rel_absrel:.4f} | {abs_absrel:.4f} | {abs_absrel - rel_absrel:+.4f} |\n")
            report.append(f"| RMSE (mm) | {rel_rmse:.2f} | {abs_rmse:.2f} | {abs_rmse - rel_rmse:+.2f} |\n")
            report.append("\n")
        
        
        with open(self.output_dir / 'evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(''.join(report))
        
        print(f"✓ 텍스트 리포트 저장: {self.output_dir / 'evaluation_report.md'}")
    
    def generate_report(self):
        """종합 리포트 생성"""
        print("\n" + "=" * 60)
        print("평가 리포트 생성 중...")
        print("=" * 60 + "\n")
        
        # 차트 생성
        self.plot_metric_comparison()
        self.plot_distance_analysis()
        self.plot_confidence_analysis()
        
        # 테이블 생성
        self.generate_summary_table()
        
        # 텍스트 리포트
        self.generate_text_report()
        
        # CSV 저장
        self.save_overall_metrics_csv()
        self.save_distance_metrics_csv()
        
        # 출력 파일 목록
        print("\n" + "=" * 60)
        print("생성된 파일 목록:")
        print("=" * 60)
        
        output_files = [
            'delta_accuracy_comparison.png',
            'distance_analysis.png',
            'confidence_analysis.png',
            'summary_table.csv',
            'evaluation_report.md',
            'overall_metrics.csv',
            'distance_metrics_detailed.csv',
        ]
        
        for f in output_files:
            path = self.output_dir / f
            if path.exists():
                size = path.stat().st_size / 1024
                print(f"  ✓ {f} ({size:.1f} KB)")
        
        print("\n모든 리포트가 생성되었습니다!")
        print(f"출력 디렉토리: {self.output_dir}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="평가 결과 시각화 및 리포트 생성")
    parser.add_argument(
        "--results_path",
        type=str,
        default="./evaluation_results/move_test_10/evaluation_results.json",
        help="evaluation_results.json 파일 경로"
    )
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.results_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {args.results_path}")
        return
    
    visualizer = EvaluationVisualizer(args.results_path)
    visualizer.generate_report()


if __name__ == "__main__":
    main()