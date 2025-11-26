# Depth Anything v3 평가 가이드

이 가이드는 Depth Anything v3 모델의 깊이 추정 결과를 평가하고 시각화하는 전체 워크플로우를 설명합니다.

## 📋 목차

1. [개요](#개요)
2. [워크플로우](#워크플로우)
3. [스크립트 설명](#스크립트-설명)
4. [사용 예제](#사용-예제)
5. [출력 파일](#출력-파일)
6. [평가 메트릭](#평가-메트릭)

---

## 개요

이 프로젝트는 Depth Anything v3의 두 가지 모델을 평가합니다:

- **DA3-Mono (Monocular)**: 상대 깊이 추정 모델
- **DA3-Metric**: 절대 깊이 추정 모델

평가는 ZED 카메라의 깊이 맵을 Ground Truth로 사용하여 수행됩니다.

---

## 워크플로우

```
1. RGB 이미지 → save_depth.py → Depth 추정 결과 저장
2. Depth 추정 결과 → evaluate_depth_models.py → 평가 수행
3. 평가 결과 → visualize_evaluation.py → 시각화 및 리포트 생성
```

---

## 스크립트 설명

### 1. `save_depth.py` - Depth 추정 및 저장

RGB 이미지에서 깊이를 추정하고 결과를 저장하는 스크립트입니다.

#### 주요 기능
- Depth Anything v3의 두 모델(Monocular, Metric)을 사용하여 깊이 추정
- 결과를 NumPy 배열(.npy) 및 시각화 이미지(.png) 형식으로 저장
- 배치 처리 지원 (단일 이미지 또는 디렉토리)

#### 사용법

```bash
python save_depth.py \
    --image ./images/ \
    --output_mono ./depth_output_rel/move \
    --output_metric ./depth_output_abs/move \
    --model_mono depth-anything/da3mono-large \
    --model_metric depth-anything/da3metric-large \
    --format both \
    --device cuda
```

#### 인자 설명
- `--image`: 입력 이미지 경로 (파일 또는 디렉토리)
- `--output_mono`: Monocular 모델 출력 디렉토리
- `--output_metric`: Metric 모델 출력 디렉토리
- `--model_mono`: Monocular 모델 이름 (Hugging Face Hub)
- `--model_metric`: Metric 모델 이름 (Hugging Face Hub)
- `--format`: 저장 형식 (`npy`, `png`, `both`)
- `--device`: 사용할 디바이스 (`cuda` 또는 `cpu`)

#### 출력 구조
```
depth_output_rel/
└── move/
    ├── depth_npy/          # NumPy 배열 (.npy)
    │   ├── 000000_depth.npy
    │   └── ...
    └── depth_visualization/  # 시각화 이미지 (.png)
        ├── 000000_depth.png
        └── ...

depth_output_abs/
└── move/
    ├── depth_npy/
    └── depth_visualization/
```

---

### 2. `evaluate_depth_models.py` - 모델 평가

추정된 깊이 맵을 Ground Truth와 비교하여 성능을 평가하는 스크립트입니다.

#### 주요 기능
- ZED 깊이 맵을 Ground Truth로 사용
- Monocular 및 Metric 모델의 성능 평가
- 거리별 성능 분석 (0-1m, 1-2m, 2-5m, 5-10m, 10-20m)
- 다양한 평가 메트릭 계산 (AbsRel, RMSE, MAE, Delta accuracy, SILog, Spearman 등)
- 결과를 JSON 형식으로 저장

#### 사용법

```bash
python evaluate_depth_models.py \
    --zed_dir ./depth_output_zed/move \
    --rel_dir ./depth_output_rel/move \
    --abs_dir ./depth_output_abs/move \
    --output_dir ./evaluation_results/move \
    --confidence_threshold 0.0 \
    --max_distance 20000.0 \
    --min_distance 200.0
```

#### 인자 설명
- `--zed_dir`: ZED 깊이 맵 디렉토리 (Ground Truth)
- `--rel_dir`: Monocular 모델 결과 디렉토리
- `--abs_dir`: Metric 모델 결과 디렉토리
- `--output_dir`: 평가 결과 저장 디렉토리
- `--confidence_threshold`: ZED confidence 임계값 (0-100)
- `--max_distance`: 최대 거리 (mm, 기본값: 20000.0)
- `--min_distance`: 최소 거리 (mm, 기본값: 200.0)

#### 출력 파일
- `evaluation_results.json`: 전체 평가 결과 (JSON 형식)

#### 평가 메트릭

**전체 이미지 메트릭:**
- **AbsRel**: Absolute Relative Error
- **RMSE**: Root Mean Squared Error (mm)
- **MAE**: Mean Absolute Error (mm)
- **RMSE_log**: RMSE in log space
- **δ1, δ2, δ3**: Delta accuracy (threshold: 1.25, 1.25², 1.25³)
- **SILog**: Scale-Invariant Logarithmic Error
- **Spearman**: Spearman rank correlation

**거리별 메트릭:**
- 각 거리 범위(0-1m, 1-2m, 2-5m, 5-10m, 10-20m)에 대해 위 메트릭들을 계산

**특징:**
- Monocular 모델: Alignment(scale + shift) 후 평가
- Metric 모델: 직접 비교 (alignment 없음)

---

### 3. `visualize_evaluation.py` - 결과 시각화 및 리포트 생성

평가 결과를 시각화하고 종합 리포트를 생성하는 스크립트입니다.

#### 주요 기능
- Delta Accuracy 비교 차트 생성
- 거리별 성능 분석 차트 생성
- 요약 테이블 및 상세 CSV 파일 생성
- 마크다운 형식의 평가 리포트 생성

#### 사용법

```bash
python visualize_evaluation.py \
    --results_path ./evaluation_results/move/evaluation_results.json
```

#### 인자 설명
- `--results_path`: `evaluate_depth_models.py`에서 생성된 JSON 파일 경로

#### 출력 파일

1. **`delta_accuracy_comparison.png`**
   - δ1, δ2, δ3 정확도 비교 차트
   - Monocular vs Metric 모델 비교

2. **`distance_analysis.png`**
   - 거리별 성능 분석 차트 (6개 서브플롯)
   - RMSE, AbsRel, Delta-1, RMSE & MAE, SILog 비교

3. **`summary_table.csv`**
   - 전체 메트릭 요약 테이블
   - Monocular 및 Metric 모델의 주요 메트릭

4. **`overall_metrics.csv`**
   - 전체 메트릭 상세 정보 (mean, std 포함)
   - 거리 구분 없이 전체 이미지 기준

5. **`distance_metrics_detailed.csv`**
   - 거리별 모든 메트릭 상세 정보
   - Long format (거리 범위 × 모델 × 메트릭)
   - 모든 메트릭의 mean, std, valid_pixels 포함

6. **`evaluation_report.md`**
   - 마크다운 형식의 종합 평가 리포트
   - 모델별 성능 요약 및 비교

---

## 사용 예제

### 전체 워크플로우 예제

```bash
# 1. RGB 이미지에서 깊이 추정
python save_depth.py \
    --image ./images/move/ \
    --output_mono ./depth_output_rel/move \
    --output_metric ./depth_output_abs/move \
    --format npy

# 2. 평가 수행
python evaluate_depth_models.py \
    --zed_dir ./depth_output_zed/move \
    --rel_dir ./depth_output_rel/move \
    --abs_dir ./depth_output_abs/move \
    --output_dir ./evaluation_results/move

# 3. 결과 시각화
python visualize_evaluation.py \
    --results_path ./evaluation_results/move/evaluation_results.json
```

---

## 출력 파일

### 평가 결과 디렉토리 구조

```
evaluation_results/
└── move/
    ├── evaluation_results.json          # 원본 평가 결과 (JSON)
    ├── delta_accuracy_comparison.png    # Delta Accuracy 비교 차트
    ├── distance_analysis.png            # 거리별 분석 차트
    ├── summary_table.csv                # 요약 테이블
    ├── overall_metrics.csv               # 전체 메트릭
    ├── distance_metrics_detailed.csv    # 거리별 상세 메트릭
    └── evaluation_report.md              # 평가 리포트
```

---

## 평가 메트릭 상세 설명

### 1. AbsRel (Absolute Relative Error)
```
AbsRel = mean(|pred - gt| / gt)
```
- 상대 오차를 측정하는 메트릭
- 값이 낮을수록 좋음
- 스케일에 덜 의존적

### 2. RMSE (Root Mean Squared Error)
```
RMSE = sqrt(mean((pred - gt)²))
```
- 절대 오차를 측정하는 메트릭
- 단위: mm
- 값이 낮을수록 좋음

### 3. MAE (Mean Absolute Error)
```
MAE = mean(|pred - gt|)
```
- 절대 오차의 평균
- 단위: mm
- 값이 낮을수록 좋음

### 4. Delta Accuracy (δ1, δ2, δ3)
```
δ = max(pred/gt, gt/pred) < threshold
Delta Accuracy = mean(δ)
```
- 정확도 메트릭
- threshold: 1.25 (δ1), 1.25² (δ2), 1.25³ (δ3)
- 값이 높을수록 좋음 (0-1 범위)

### 5. SILog (Scale-Invariant Logarithmic Error)
```
SILog = sqrt(mean(log_diff²) - mean(log_diff)²)
```
- 스케일 불변 로그 오차
- 값이 낮을수록 좋음
- 전체 이미지에 대해 계산

### 6. Spearman Correlation
- 순위 상관계수
- 값이 높을수록 좋음 (-1 ~ 1 범위)
- 예측과 실제 깊이의 순서 일치도를 측정

---

## 주의사항

1. **파일 이름 규칙**
   - ZED 파일: `000000.npy`, `000001.npy`, ...
   - 추정 결과: `000000_depth.npy`, `000001_depth.npy`, ...
   - 파일 이름의 숫자 부분이 일치해야 합니다

2. **단위**
   - ZED 깊이 맵: mm 단위
   - Metric 모델 출력: 자동으로 mm 단위로 변환 (m 단위인 경우)

3. **거리 범위**
   - 기본 거리 범위: 0-1m, 1-2m, 2-5m, 5-10m, 10-20m
   - 각 범위에 최소 100개 픽셀이 있어야 평가됨

4. **Confidence Threshold**
   - ZED confidence map이 있는 경우 사용
   - 없으면 모든 픽셀 사용

---

## 문제 해결

### Q: "파일을 찾을 수 없습니다" 오류
- 파일 이름이 올바른지 확인
- 디렉토리 경로가 올바른지 확인
- 파일 확장자(.npy) 확인

### Q: 거리별 메트릭이 비어있음
- 해당 거리 범위에 충분한 픽셀이 있는지 확인 (최소 100개)
- valid_mask가 올바르게 생성되었는지 확인

### Q: 평가 결과가 예상과 다름
- 단위 확인 (mm vs m)
- confidence threshold 확인
- min_distance, max_distance 설정 확인

---

## 참고 자료

- [Depth Anything v3](https://github.com/DepthAnything/Depth-Anything-V3)
- [ZED Camera Documentation](https://www.stereolabs.com/docs/)

---

## 라이선스

이 코드는 프로젝트의 라이선스를 따릅니다.

