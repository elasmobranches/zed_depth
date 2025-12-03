"""
GPU 사용률 측정 및 CSV 로깅 (jtop 사용)

사용법:
    # 기본 사용 (Enter로 시작, Ctrl+C로 종료)
    python3 measure_gpu.py
    
    # 시나리오 이름 지정
    python3 measure_gpu.py --name "ZED Neural+"
    
    # 여러 시나리오 측정
    python3 measure_gpu.py --name "Idle"
    python3 measure_gpu.py --name "ZED Neural+"
    python3 measure_gpu.py --name "DA3-Metric"
"""

import time
import csv
from pathlib import Path
import argparse
from datetime import datetime

try:
    from jtop import jtop
except ImportError:
    print("❌ jtop이 설치되지 않았습니다.")
    print("설치: sudo -H pip3 install -U jetson-stats")
    print("설치 후 재부팅: sudo reboot")
    exit(1)


def measure_gpu(scenario_name: str, output_file: str = "gpu_measurements.csv"):
    """
    GPU 사용률 측정
    
    Args:
        scenario_name: 측정 시나리오 이름
        output_file: 저장할 CSV 파일명
    """
    print(f"\n{'='*60}")
    print(f"GPU 사용률 측정: {scenario_name}")
    print(f"{'='*60}\n")
    
    # 파일 존재 여부 확인 (헤더 작성용)
    csv_path = Path(output_file)
    file_exists = csv_path.exists()
    
    input(f"측정할 프로그램을 실행한 후 Enter를 눌러 측정을 시작하세요...")
    
    print(f"\n측정 중... (종료하려면 Ctrl+C)\n")
    
    gpu_samples = []
    start_time = time.time()
    
    try:
        with jtop() as jetson:
            while True:
                try:
                    gpu_usage = jetson.stats['GPU']
                    gpu_samples.append(gpu_usage)
                except:
                    gpu_usage = 0
                
                elapsed = time.time() - start_time
                print(f"\r측정 시간: {elapsed:.1f}초 | GPU: {gpu_usage:.1f}% | "
                      f"평균: {sum(gpu_samples)/len(gpu_samples):.1f}%", end='')
                
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        total_time = time.time() - start_time
        print("\n\n✓ 측정 종료\n")
    
    # 통계 계산
    valid_samples = [s for s in gpu_samples if s is not None]
    
    if valid_samples:
        avg_gpu = sum(valid_samples) / len(valid_samples)
        max_gpu = max(valid_samples)
        min_gpu = min(valid_samples)
        
        # 결과 출력
        print(f"{'='*60}")
        print(f"측정 결과")
        print(f"{'='*60}")
        print(f"시나리오: {scenario_name}")
        print(f"측정 시간: {total_time:.1f}초")
        print(f"샘플 수: {len(valid_samples)}개")
        print(f"평균 GPU: {avg_gpu:.1f}%")
        print(f"최대 GPU: {max_gpu:.1f}%")
        print(f"최소 GPU: {min_gpu:.1f}%")
        print(f"{'='*60}\n")
        
        # CSV 저장
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 헤더 (파일이 없을 때만)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'scenario', 'duration_sec', 'num_samples',
                    'avg_gpu_percent', 'max_gpu_percent', 'min_gpu_percent'
                ])
            
            # 데이터
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                scenario_name,
                f"{total_time:.1f}",
                len(valid_samples),
                f"{avg_gpu:.1f}",
                f"{max_gpu:.1f}",
                f"{min_gpu:.1f}"
            ])
        
        print(f"✓ 결과 저장: {csv_path}\n")
    else:
        print("⚠ 유효한 GPU 샘플이 없습니다.\n")


def main():
    parser = argparse.ArgumentParser(
        description="GPU 사용률 측정 및 CSV 로깅",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용
  python3 measure_gpu.py --name "ZED Neural+"
  
  # 여러 시나리오 측정
  python3 measure_gpu.py --name "Idle"
  python3 measure_gpu.py --name "ZED Neural+"
  python3 measure_gpu.py --name "DA3-Metric"
  
  # 결과 확인
  cat gpu_measurements.csv
        """
    )
    
    parser.add_argument('-n', '--name', type=str, default='Measurement',
                        help='시나리오 이름 (예: "ZED Neural+", "DA3-Metric")')
    parser.add_argument('-o', '--output', type=str, default='gpu_measurements.csv',
                        help='출력 CSV 파일명 (기본값: gpu_measurements.csv)')
    
    args = parser.parse_args()
    
    measure_gpu(args.name, args.output)


if __name__ == "__main__":
    main()

