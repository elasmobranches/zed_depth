# ZED 2i Stereo & Depth Anything V3 평가 파이프라인

## 개요

이 프로젝트는 **Depth Anything v3**의 두 가지 모델을 평가합니다:

- **DA3-Mono (Monocular)**: 상대 깊이 추정 모델
- **DA3-Metric**: 절대 깊이 추정 모델

평가는 **ZED 카메라의 깊이 맵을 Ground Truth**로 사용하여 수행됩니다.


### 1. `extract_svo.py` - SVO 파일에서 프레임 추출

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
python extract_frame.py
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
python extract_svo.py \
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

---

## 3. `save_depth.py` - Depth 추정 및 저장

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


### 4. `analyze_confidence.py` - Confidence Map 분석 

ZED Confidence map을 분석하고 시각화하는 스크립트입니다.

#### 주요 기능
- Confidence map 통계 계산 (min, max, mean, median, std)
- 임계값별 유효 픽셀 비율 계산
- Confidence 히스토그램 및 히트맵 시각화
- JSON 형식으로 통계 저장
- Confidence로 인한 filtering을 통해 유효한 픽셀끼리만 비교 가능합니다.

### 5. `evaluate_depth_models.py` - 모델 평가

추정된 깊이 맵을 Ground Truth와 비교하여 성능을 평가하는 스크립트입니다.

#### 주요 기능
- ZED 깊이 맵을 Ground Truth로 사용
- Monocular 및 Metric 모델의 성능 평가
- 거리별 성능 분석 (0-1m, 1-2m, 2-5m, 5-10m, 10-20m)
- 다양한 평가 메트릭 계산 (AbsRel, RMSE, MAE, Delta accuracy, SILog, Spearman 등)

### 6. `visualize_evaluation.py` - 결과 시각화 및 리포트 생성

평가 결과를 시각화하고 종합 리포트를 생성하는 스크립트입니다.

#### 주요 기능
- 요약 테이블 및 상세 CSV 파일 생성
- 마크다운 형식의 평가 리포트 생성
- 거리별 성능 분석 차트 생성
- Delta Accuracy 비교 차트 생성
