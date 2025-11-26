"""
평가 결과 시각화 및 리포트 생성
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List


class EvaluationVisualizer:
    """평가 결과 시각화 클래스"""
    
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
        if not self.results:
            print("Warning: Results file is empty!")
        elif 'relative' not in self.results and 'absolute' not in self.results:
            print("Warning: No evaluation results found in the file!")
        
        # 스타일 설정
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                plt.style.use('default')
        sns.set_palette("husl")
        
        # 한글 폰트 경고 방지: 영어로 설정하거나 폰트 설정
        import matplotlib
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        # 한글 경고 무시
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    def _get_nested_value(self, d: Dict, key_path: str, default: float = 0) -> float:
        """
        중첩된 딕셔너리에서 값 가져오기
        JSON이 평탄화되어 있으므로 (예: "after_alignment.abs_rel": {"mean": ...})
        key_path가 "after_alignment.abs_rel.mean" 형태일 때 올바르게 처리
        """
        keys = key_path.split('.')
        
        # 마지막 키가 'mean', 'std' 등인지 확인
        if len(keys) >= 2 and keys[-1] in ['mean', 'std', 'min', 'max']:
            # "after_alignment.abs_rel.mean" -> "after_alignment.abs_rel"과 "mean"으로 분리
            main_key = '.'.join(keys[:-1])  # "after_alignment.abs_rel"
            sub_key = keys[-1]  # "mean"
            
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
        return float(val) if isinstance(val, (int, float)) else default
    
    def plot_metric_comparison(self):
        """주요 메트릭 비교 차트"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 데이터 준비
        metrics = ['AbsRel', 'RMSE (mm)', 'δ1', 'SILog']
        rel_values = []
        abs_values = []
        
        if 'relative' in self.results:
            rel = self.results['relative']
            rel_values = [
                self._get_nested_value(rel, 'after_alignment.abs_rel.mean', 0),
                self._get_nested_value(rel, 'after_alignment.rmse.mean', 0),
                self._get_nested_value(rel, 'after_alignment.delta_1.mean', 0),
                self._get_nested_value(rel, 'scale_invariant.silog.mean', 0)
            ]
        
        if 'absolute' in self.results:
            abs_ = self.results['absolute']
            abs_values = [
                self._get_nested_value(abs_, 'direct_comparison.abs_rel.mean', 0),
                self._get_nested_value(abs_, 'direct_comparison.rmse.mean', 0),
                self._get_nested_value(abs_, 'direct_comparison.delta_1.mean', 0),
                self._get_nested_value(abs_, 'scale_invariant.silog.mean', 0)
            ]
        
        # Bar plot
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, rel_values, width, label='DA3-Mono (Relative)', alpha=0.8)
        axes[0, 0].bar(x + width/2, abs_values, width, label='DA3-Metric (Absolute)', alpha=0.8)
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Metric Comparison', fontsize=12)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
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
            
            x_delta = np.arange(len(deltas))
            axes[0, 1].bar(x_delta - width/2, rel_deltas, width, label='DA3-Mono', alpha=0.8)
            axes[0, 1].bar(x_delta + width/2, abs_deltas, width, label='DA3-Metric', alpha=0.8)
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Delta Accuracy Comparison', fontsize=12)
            axes[0, 1].set_xticks(x_delta)
            axes[0, 1].set_xticklabels(deltas)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1])
        
        # 거리별 성능 (상대 모델)
        if 'relative' in self.results:
            rel = self.results['relative']
            # distance_analysis 데이터 추출
            distance_ranges = ['0.0-1.0m', '1.0-2.0m', '2.0-5.0m', '5.0-10.0m', '10.0-20.0m']
            rel_rmse = []
            rel_abs_rel = []
            
            for dist_range in distance_ranges:
                key_rmse = f'distance_analysis.{dist_range}.rmse'
                key_abs_rel = f'distance_analysis.{dist_range}.abs_rel'
                rel_rmse.append(self._get_nested_value(rel, key_rmse + '.mean', 0))
                rel_abs_rel.append(self._get_nested_value(rel, key_abs_rel + '.mean', 0))
            
            # RMSE 차트
            x_dist = np.arange(len(distance_ranges))
            axes[1, 0].bar(x_dist, rel_rmse, alpha=0.7, color='steelblue')
            axes[1, 0].set_xlabel('Distance Range')
            axes[1, 0].set_ylabel('RMSE (mm)')
            axes[1, 0].set_title('Distance-based RMSE (Relative Model)', fontsize=12)
            axes[1, 0].set_xticks(x_dist)
            axes[1, 0].set_xticklabels(distance_ranges, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Scale drift (절대 모델)
        if 'absolute' in self.results:
            abs_ = self.results['absolute']
            if 'scale_drift' in abs_:
                scale_drift = abs_['scale_drift']
                if scale_drift:
                    distances = [s['distance_range'][0] / 1000 for s in scale_drift]
                    scales = [s['scale_factor'] for s in scale_drift]
                    
                    axes[1, 1].plot(distances, scales, 'o-', linewidth=2, markersize=8)
                    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Ideal (1.0)')
                    axes[1, 1].set_xlabel('Distance (m)')
                    axes[1, 1].set_ylabel('Scale Factor')
                    axes[1, 1].set_title('Scale Drift Analysis (Absolute Model)', fontsize=12)
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
        print(f"메트릭 비교 차트 저장: {self.output_dir / 'metric_comparison.png'}")
        plt.close()
    
    def plot_distance_analysis(self):
        """거리별 성능 분석 차트"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        distance_ranges = ['0.0-1.0m', '1.0-2.0m', '2.0-5.0m', '5.0-10.0m', '10.0-20.0m']
        x_dist = np.arange(len(distance_ranges))
        
        # 상대 모델 거리별 분석
        if 'relative' in self.results:
            rel = self.results['relative']
            
            rel_rmse = []
            rel_abs_rel = []
            rel_delta_1 = []
            
            for dist_range in distance_ranges:
                rel_rmse.append(self._get_nested_value(rel, f'distance_analysis.{dist_range}.rmse.mean', 0))
                rel_abs_rel.append(self._get_nested_value(rel, f'distance_analysis.{dist_range}.abs_rel.mean', 0))
                rel_delta_1.append(self._get_nested_value(rel, f'distance_analysis.{dist_range}.delta_1.mean', 0))
            
            # RMSE
            axes[0, 0].bar(x_dist, rel_rmse, alpha=0.7, color='steelblue', label='Relative')
            axes[0, 0].set_xlabel('Distance Range')
            axes[0, 0].set_ylabel('RMSE (mm)')
            axes[0, 0].set_title('RMSE by Distance (Relative Model)', fontsize=12)
            axes[0, 0].set_xticks(x_dist)
            axes[0, 0].set_xticklabels(distance_ranges, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # AbsRel
            axes[0, 1].bar(x_dist, rel_abs_rel, alpha=0.7, color='coral', label='Relative')
            axes[0, 1].set_xlabel('Distance Range')
            axes[0, 1].set_ylabel('AbsRel')
            axes[0, 1].set_title('AbsRel by Distance (Relative Model)', fontsize=12)
            axes[0, 1].set_xticks(x_dist)
            axes[0, 1].set_xticklabels(distance_ranges, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 절대 모델 거리별 분석
        if 'absolute' in self.results:
            abs_ = self.results['absolute']
            
            abs_rmse = []
            abs_abs_rel = []
            abs_delta_1 = []
            
            for dist_range in distance_ranges:
                abs_rmse.append(self._get_nested_value(abs_, f'distance_analysis.{dist_range}.rmse.mean', 0))
                abs_abs_rel.append(self._get_nested_value(abs_, f'distance_analysis.{dist_range}.abs_rel.mean', 0))
                abs_delta_1.append(self._get_nested_value(abs_, f'distance_analysis.{dist_range}.delta_1.mean', 0))
            
            # RMSE
            axes[1, 0].bar(x_dist, abs_rmse, alpha=0.7, color='steelblue', label='Absolute')
            axes[1, 0].set_xlabel('Distance Range')
            axes[1, 0].set_ylabel('RMSE (mm)')
            axes[1, 0].set_title('RMSE by Distance (Absolute Model)', fontsize=12)
            axes[1, 0].set_xticks(x_dist)
            axes[1, 0].set_xticklabels(distance_ranges, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # AbsRel
            axes[1, 1].bar(x_dist, abs_abs_rel, alpha=0.7, color='coral', label='Absolute')
            axes[1, 1].set_xlabel('Distance Range')
            axes[1, 1].set_ylabel('AbsRel')
            axes[1, 1].set_title('AbsRel by Distance (Absolute Model)', fontsize=12)
            axes[1, 1].set_xticks(x_dist)
            axes[1, 1].set_xticklabels(distance_ranges, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"거리별 분석 차트 저장: {self.output_dir / 'distance_analysis.png'}")
        plt.close()
    
    def generate_summary_table(self):
        """요약 테이블 생성"""
        data = []
        
        if 'relative' in self.results:
            rel = self.results['relative']
            data.append({
                'Model': 'DA3-Mono (Relative)',
                'AbsRel': f"{self._get_nested_value(rel, 'after_alignment.abs_rel.mean', 0):.4f}",
                'RMSE (mm)': f"{self._get_nested_value(rel, 'after_alignment.rmse.mean', 0):.2f}",
                'δ1': f"{self._get_nested_value(rel, 'after_alignment.delta_1.mean', 0):.4f}",
                'Spearman': f"{self._get_nested_value(rel, 'scale_invariant.spearman.mean', 0):.4f}",
                'SILog': f"{self._get_nested_value(rel, 'scale_invariant.silog.mean', 0):.4f}"
            })
        
        if 'absolute' in self.results:
            abs_ = self.results['absolute']
            data.append({
                'Model': 'DA3-Metric (Absolute)',
                'AbsRel': f"{self._get_nested_value(abs_, 'direct_comparison.abs_rel.mean', 0):.4f}",
                'RMSE (mm)': f"{self._get_nested_value(abs_, 'direct_comparison.rmse.mean', 0):.2f}",
                'δ1': f"{self._get_nested_value(abs_, 'direct_comparison.delta_1.mean', 0):.4f}",
                'Spearman': (
                    f"{self._get_nested_value(abs_, 'scale_invariant.spearman.mean', 0):.4f}"
                    if self._get_nested_value(abs_, 'scale_invariant.spearman.mean', None) is not None
                    else '-'
                ),
                'SILog': (
                    f"{self._get_nested_value(abs_, 'scale_invariant.silog.mean', 0):.4f}"
                    if self._get_nested_value(abs_, 'scale_invariant.silog.mean', None) is not None
                    else '-'
                )
            })
        
        if not data:
            print("경고: 표시할 데이터가 없습니다.")
            return
        
        df = pd.DataFrame(data)
        
        # LaTeX 테이블로 저장
        try:
            with open(self.output_dir / 'summary_table.tex', 'w', encoding='utf-8') as f:
                f.write(df.to_latex(index=False, float_format="%.4f"))
        except Exception as e:
            print(f"LaTeX 테이블 저장 실패: {e}")
        
        # CSV로 저장
        df.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        # Markdown으로 저장
        try:
            with open(self.output_dir / 'summary_table.md', 'w', encoding='utf-8') as f:
                f.write("# Evaluation Results Summary\n\n")
                # to_markdown은 pandas 1.0.0+ 에서 사용 가능하지만 tabulate 필요
                try:
                    if hasattr(df, 'to_markdown'):
                        f.write(df.to_markdown(index=False))
                    else:
                        raise AttributeError("to_markdown not available")
                except (AttributeError, ImportError):
                    # 대체 방법: 간단한 마크다운 테이블 수동 생성
                    f.write("| " + " | ".join(df.columns) + " |\n")
                    f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
                    for _, row in df.iterrows():
                        f.write("| " + " | ".join(str(val) for val in row) + " |\n")
        except Exception as e:
            print(f"Warning: Markdown table save failed: {e}")
        
        print(f"\n요약 테이블 저장:")
        print(f"  - {self.output_dir / 'summary_table.csv'}")
        print(f"  - {self.output_dir / 'summary_table.md'}")
        print(f"  - {self.output_dir / 'summary_table.tex'}")
        
        print("\n" + df.to_string(index=False))
    
    def generate_report(self):
        """종합 리포트 생성"""
        print("\n" + "=" * 60)
        print("평가 리포트 생성 중...")
        print("=" * 60)
        
        # 메트릭 비교 차트
        self.plot_metric_comparison()
        
        # 거리별 분석
        self.plot_distance_analysis()
        
        # 요약 테이블
        self.generate_summary_table()
        
        # 텍스트 리포트
        self.generate_text_report()
        
        print("\n모든 리포트가 생성되었습니다!")
    
    def generate_text_report(self):
        """텍스트 리포트 생성"""
        report = []
        report.append("# Depth 모델 평가 리포트\n")
        report.append(f"생성 시간: {pd.Timestamp.now()}\n\n")
        
        report.append("## 1. 상대 깊이 모델 (DA3-Mono)\n")
        if 'relative' in self.results:
            rel = self.results['relative']
            report.append("### 주요 메트릭\n")
            report.append(f"- **AbsRel**: {self._get_nested_value(rel, 'after_alignment.abs_rel.mean', 0):.4f}\n")
            report.append(f"- **RMSE**: {self._get_nested_value(rel, 'after_alignment.rmse.mean', 0):.2f} mm\n")
            report.append(f"- **δ1**: {self._get_nested_value(rel, 'after_alignment.delta_1.mean', 0):.4f}\n")
            report.append(f"- **Spearman**: {self._get_nested_value(rel, 'scale_invariant.spearman.mean', 0):.4f}\n")
            report.append(f"- **SILog**: {self._get_nested_value(rel, 'scale_invariant.silog.mean', 0):.4f}\n\n")
        
        report.append("## 2. 절대 깊이 모델 (DA3-Metric)\n")
        if 'absolute' in self.results:
            abs_ = self.results['absolute']
            report.append("### 주요 메트릭\n")
            report.append(f"- **AbsRel**: {self._get_nested_value(abs_, 'direct_comparison.abs_rel.mean', 0):.4f}\n")
            report.append(f"- **RMSE**: {self._get_nested_value(abs_, 'direct_comparison.rmse.mean', 0):.2f} mm\n")
            report.append(f"- **δ1**: {self._get_nested_value(abs_, 'direct_comparison.delta_1.mean', 0):.4f}\n\n")
            abs_spearman = self._get_nested_value(abs_, 'scale_invariant.spearman.mean', None)
            abs_silog = self._get_nested_value(abs_, 'scale_invariant.silog.mean', None)
            if abs_spearman is not None or abs_silog is not None:
                report.append("### Scale-invariant 메트릭\n")
                if abs_spearman is not None:
                    report.append(f"- **Spearman**: {abs_spearman:.4f}\n")
                if abs_silog is not None:
                    report.append(f"- **SILog**: {abs_silog:.4f}\n")
                report.append("\n")
        
        report.append("## 3. 모델 비교\n")
        if 'comparison' in self.results:
            comp = self.results['comparison']['summary']
            report.append("### 성능 비교\n")
            
            def format_comp_value(val):
                """비교 값 포맷팅"""
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
            
            report.append(f"- 상대 모델 AbsRel: {format_comp_value(rel_abs_rel)}\n")
            report.append(f"- 절대 모델 AbsRel: {format_comp_value(abs_abs_rel)}\n")
            if rel_rmse is not None or abs_rmse is not None:
                report.append(f"- 상대 모델 RMSE: {format_comp_value(rel_rmse)} mm\n")
                report.append(f"- 절대 모델 RMSE: {format_comp_value(abs_rmse)} mm\n")
            report.append("\n")
        
        report.append("## 4. 권장사항\n")
        report.append("### DA3-Mono (상대 깊이) 권장 상황:\n")
        report.append("- 상대적 깊이 관계만 필요한 경우\n")
        report.append("- 엣지와 세부 디테일이 중요한 경우\n")
        report.append("- 후처리로 스케일 조정이 가능한 경우\n\n")
        
        report.append("### DA3-Metric (절대 깊이) 권장 상황:\n")
        report.append("- 절대적인 거리 측정이 필요한 경우\n")
        report.append("- 로봇 네비게이션, 자율주행 등\n")
        report.append("- Ground truth 없이 바로 사용해야 하는 경우\n")
        
        with open(self.output_dir / 'evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(''.join(report))
        
        print(f"텍스트 리포트 저장: {self.output_dir / 'evaluation_report.md'}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="평가 결과 시각화")
    parser.add_argument(
        "--results_path",
        type=str,
        default="./evaluation_results/not_move/evaluation_results.json",
        help="evaluation_results.json 파일 경로"
    )
    
    args = parser.parse_args()
    
    visualizer = EvaluationVisualizer(args.results_path)
    visualizer.generate_report()


if __name__ == "__main__":
    main()

