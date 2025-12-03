# ZED 2i Stereo & Depth Anything V3 평가 파이프라인

## 개요

이 프로젝트는 **Depth Anything v3**의 두 가지 모델을 평가합니다:

- **DA3-Mono (Monocular)**: 상대 깊이 추정 모델
- **DA3-Metric**: 절대 깊이 추정 모델
huggingface 주소 `https://huggingface.co/depth-anything`
평가는 **ZED 카메라의 깊이 맵을 Ground Truth**로 사용하여 수행됩니다.


### 1. `extract_svo.py` - SVO 파일에서 녹화

ZED 카메라를 사용하여 실시간으로 영상을 녹화하고 SVO 파일로 저장하는 스크립트입니다.

#### 주요 기능
- ZED 카메라 실시간 녹화
- Depth 데이터 포함 SVO2 형식 저장
- NEURAL_PLUS depth 모드 사용
- 깊이 범위: 200mm ~ 20,000mm (20m)
- 해상도: 1920×1080 (HD1080), 30fps
- 실시간 depth 유효성 검증

#### 중요 설정
```python
init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
init_params.coordinate_units = sl.UNIT.MILLIMETER
init_params.depth_minimum_distance = 200  # 20cm
init_params.depth_maximum_distance = 20000  # 20m
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
```

#### 사용법

녹화 시작:
```bash
python extract_svo.py
```

- Ctrl-C로 종료
- SVO 파일은 `/home/cv_test/Desktop/MCT_for_ChamDog/depth_capture/recordings/`에 저장
- 파일명 형식: `zed_YYYYMMDD_HHMMSS.svo2`

### 2. `extract_frame.py` - SVO 파일에서 프레임 추출

#### 주요 기능
- SVO 파일에서 프레임 추출
- RGB, Depth, Confidence 동시 추출
- 샘플링 지원 (N프레임당 1개 추출)
- Depth 유효성 실시간 검증
- NumPy 배열(.npy) 및 시각화 이미지(.png) 저장

#### 사용법

```bash
python extract_frame.py \
    --svo_path ./recordings/zed_20250115_143022.svo2 \
    --output_dir ./depth_output_zed/move \
    --sample_rate 10
```

#### 인자 설명
- `--svo_path`: 입력 SVO 파일 경로 (필수)
- `--output_dir`: 출력 디렉토리 (기본값: `./extracted_frames/not_move`)
- `--sample_rate`: 샘플링 비율 (기본값: 10, 10프레임당 1개 추출)

#### 중요 사항
- 녹화 시와 동일한 depth 설정 사용 필수


### 3. `evaluate_depth_models.py` - 모델 평가

추정된 깊이 맵을 Ground Truth와 비교하여 성능을 평가하는 스크립트입니다.

#### 주요 기능
- ZED 깊이 맵을 Ground Truth로 사용
- Monocular 및 Metric 모델의 성능 평가
- 거리별 성능 분석 (0-1m, 1-2m, 2-5m, 5-10m, 10-20m)
- Confidence별 성능 분석 (0-20, 20-40, 40-60, 60-80, 80-100)
- 다양한 평가 메트릭 계산 (AbsRel, RMSE, MAE, Delta 1/2/3, SILog, Spearman 등)
- Confidence threshold 기반 유효 픽셀 필터링

#### 사용법

```bash
python evaluate_depth_models.py \
    --zed_dir ./depth_output_zed/move \
    --rel_dir ./depth_output_rel/move \
    --abs_dir ./depth_output_abs/move \
    --output_dir ./evaluation_results/move_test_10 \
    --confidence_threshold 10 \
    --max_distance 20000.0 \
    --min_distance 200.0
```

#### 인자 설명
- `--zed_dir`: ZED depth 디렉토리 (필수)
- `--rel_dir`: 상대 깊이(Monocular) 결과 디렉토리 (필수)
- `--abs_dir`: 절대 깊이(Metric) 결과 디렉토리 (필수)
- `--output_dir`: 결과 저장 디렉토리 (기본값: `./evaluation_results/move_test_10`)
- `--confidence_threshold`: ZED confidence threshold (0-100, 기본값: 10)
- `--max_distance`: 최대 거리 mm (기본값: 20000.0)
- `--min_distance`: 최소 거리 mm (기본값: 200.0)

### 4. `visualize_evaluation.py` - 결과 시각화 및 리포트 생성

평가 결과를 시각화하고 종합 리포트를 생성하는 스크립트입니다.

#### 주요 기능
- 요약 테이블 및 상세 CSV 파일 생성
- 마크다운 형식의 평가 리포트 생성
- 거리별 성능 분석 차트 생성 (RMSE, AbsRel, Delta, SILog 등)
- Confidence별 성능 분석 차트 생성
- Delta Accuracy 비교 차트 생성
- 모델 간 비교 분석

#### 사용법

```bash
python visualize_evaluation.py \
    --results_path ./evaluation_results/move_test_10/evaluation_results.json
```

#### 생성되는 파일
- `evaluation_report.md`: 종합 평가 리포트
- `summary_table.csv`: 모델별 요약 테이블
- `overall_metrics.csv`: 전체 메트릭 요약
- `distance_metrics_detailed.csv`: 거리별 상세 메트릭
- `delta_accuracy_comparison.png`: Delta accuracy 비교 차트
- `distance_analysis.png`: 거리별 성능 분석 차트
- `confidence_analysis.png`: Confidence별 성능 분석 차트

## 추가 도구

### 5. `aruco_validation/` - ArUco 마커 기반 Depth 검증

ZED 카메라의 depth 정확도를 ArUco 마커를 사용하여 검증하는 도구입니다.

#### 주요 기능
- ArUco 마커 자동 검출 및 depth 측정
- 실제 거리와 ZED 측정 거리 비교
- 오차율 자동 계산 및 통계 분석
- 실시간 시각화 (오차에 따른 색상 표시)
- 측정 데이터 CSV 저장

#### 사용법

1. **마커 배치 및 설정**
   ```yaml
   # marker_config.yaml 수정
   markers:
     0: 1.0    # ID 0 = 1m
     1: 2.0    # ID 1 = 2m
     2: 3.0    # ID 2 = 3m
     3: 5.0    # ID 3 = 5m
   ```

2. **검증 실행**
   ```bash
   cd aruco_validation
   python capture_aruco.py
   ```

3. **키보드 조작**
   - `s` - 현재 프레임 저장
   - `q` - 종료

#### 결과 파일
- `results/aruco_validation_YYYYMMDD_HHMMSS.csv` - 측정 데이터
- `results/capture_YYYYMMDD_HHMMSS.jpg` - 캡처 이미지

자세한 내용은 `aruco_validation/README.md`를 참조하세요.

### 6. `tools/` - 분석 및 측정 도구

#### GPU 사용률 측정 (`measure_gpu.py`)

다양한 시나리오에서 GPU 사용률을 측정하는 도구입니다.

```bash
cd tools
python measure_gpu.py --name "Scenario Name"
# Enter 키 누른 후 측정 시작
# Ctrl+C로 종료
```

결과는 `gpu_measurements.csv`에 저장됩니다.

#### Depth 이미지 비교 (`compare_depth_images.py`)

두 개의 depth map을 ground truth와 비교하여 메트릭을 계산합니다.

```bash
cd tools
python compare_depth_images.py \
    --pred1 ../depth_output_rel/origin/depth_npy/000000_depth.npy \
    --pred2 ../depth_output_abs/origin/depth_npy/000000_depth.npy \
    --gt ../depth_output_zed/origin/depth_npy/000000.npy \
    --name1 "Monocular" \
    --name2 "Metric" \
    --output ../comparison_results
```

자세한 내용은 `tools/README.md`를 참조하세요.

### 기타

#### `save_depth.py` - Depth 추정 및 저장

RGB 이미지에서 깊이를 추정하고 결과를 저장하는 스크립트입니다.

#### 주요 기능
- Depth Anything v3의 두 모델(Monocular, Metric)을 사용하여 깊이 추정
- 결과를 NumPy 배열(.npy) 및 시각화 이미지(.png) 형식으로 저장

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


#### `confidence.py` 

각 픽셀의 confidence 값에 따른 비율 분석