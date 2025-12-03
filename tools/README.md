# Tools

## 📊 GPU 사용률 측정

### 사용법

```bash
cd tools

# 1. Idle 측정
python3 measure_gpu.py --name "Idle"
# → Enter 누르기 → 30초 대기 → Ctrl+C

# 2. ZED 측정
python3 measure_gpu.py --name "ZED Neural+"
# → ZED 프로그램 실행 → Enter → Ctrl+C

# 3. DA3 측정
python3 measure_gpu.py --name "DA3-Metric"
# → DA3 프로그램 실행 → Enter → Ctrl+C

# 4. 결과 확인
cat gpu_measurements.csv
```

### 결과 파일

`gpu_measurements.csv`:
```csv
timestamp,scenario,duration_sec,num_samples,avg_gpu_percent,max_gpu_percent,min_gpu_percent
2025-12-03 15:00:00,Idle,30.0,300,5.2,12.3,2.1
2025-12-03 15:01:00,ZED Neural+,45.2,452,65.4,78.2,52.1
2025-12-03 15:02:00,DA3-Metric,38.7,387,42.1,55.8,35.2
```

---

## 🔍 Depth 이미지 비교

두 depth map (numpy 파일) 비교 및 메트릭 계산

```bash
python3 compare_depth_images.py \
    --pred1 ../depth_output_rel/origin/depth_npy/000000_depth.npy \
    --pred2 ../depth_output_abs/origin/depth_npy/000000_depth.npy \
    --gt ../depth_output_zed/origin/depth_npy/000000.npy \
    --name1 "Monocular" \
    --name2 "Metric" \
    --output ../comparison_results
```
