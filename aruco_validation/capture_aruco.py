"""
ArUco 마커 기반 ZED Depth 검증

사용법:
    python3 capture_aruco.py
    
키보드 조작:
    s: 현재 프레임 저장
    q: 종료
    
결과:
    results/ 폴더에 CSV 및 이미지 저장
"""

import cv2
import numpy as np
import pyzed.sl as sl
import yaml
from pathlib import Path
from datetime import datetime
import csv


class ArucoDepthValidator:
    """ArUco 마커 기반 Depth 검증 클래스"""
    
    def __init__(self, config_path='marker_config.yaml'):
        """
        Args:
            config_path: 마커 설정 파일 경로
        """
        # 설정 로드
        self.config = self.load_config(config_path)
        self.marker_distances = self.config['markers']
        self.aruco_dict_name = self.config['aruco_dict']
        
        # ArUco detector 초기화
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.aruco_dict_name))
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        # 결과 저장용
        self.results = []
        
        # 출력 디렉토리
        self.output_dir = Path('results')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"✓ ArUco Detector 초기화 완료 ({self.aruco_dict_name})")
        print(f"✓ 마커 {len(self.marker_distances)}개 설정됨")
    
    def load_config(self, config_path):
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def detect_aruco(self, image):
        """
        ArUco 마커 검출
        
        Args:
            image: BGR 이미지
        
        Returns:
            dict: {marker_id: (px, py, corners)}
        """
        corners, ids, _ = self.detector.detectMarkers(image)
        
        results = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # 마커 중심 좌표 계산
                center = corners[i][0].mean(axis=0)
                px, py = int(center[0]), int(center[1])
                results[marker_id] = (px, py, corners[i][0])
        
        return results
    
    def get_depth_median(self, depth_map, px, py, window_size=5):
        """
        5x5 윈도우의 중앙값으로 depth 측정 (노이즈 제거)
        
        Args:
            depth_map: Depth 이미지
            px, py: 중심 좌표
            window_size: 윈도우 크기 (홀수)
        
        Returns:
            중앙값 depth (mm)
        """
        half = window_size // 2
        
        h, w = depth_map.shape
        y_start = max(0, py - half)
        y_end = min(h, py + half + 1)
        x_start = max(0, px - half)
        x_end = min(w, px + half + 1)
        
        window = depth_map[y_start:y_end, x_start:x_end]
        
        # 유효한 값들만 추출 (NaN, Inf 제외)
        valid_values = window[np.isfinite(window)]
        
        if len(valid_values) > 0:
            return np.median(valid_values)
        else:
            return np.nan
    
    def get_error_color(self, error_pct):
        """
        오차에 따른 색상 반환
        
        Args:
            error_pct: 오차 비율 (%)
        
        Returns:
            BGR 색상 튜플
        """
        if error_pct < 3.0:
            return (0, 255, 0)  # 녹색 (좋음)
        elif error_pct < 5.0:
            return (0, 255, 255)  # 노란색 (보통)
        else:
            return (0, 0, 255)  # 빨간색 (나쁨)
    
    def draw_marker_info(self, image, marker_id, px, py, corners, zed_depth_mm, actual_mm, error_pct):
        """
        마커 정보를 이미지에 그리기
        
        Args:
            image: BGR 이미지
            marker_id: 마커 ID
            px, py: 마커 중심
            corners: 마커 코너 좌표
            zed_depth_mm: ZED depth (mm)
            actual_mm: 실제 거리 (mm)
            error_pct: 오차 비율 (%)
        """
        color = self.get_error_color(error_pct)
        
        # 마커 외곽선 그리기
        pts = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(image, [pts], True, color, 3)
        
        # 중심점 그리기
        cv2.circle(image, (px, py), 8, color, -1)
        cv2.circle(image, (px, py), 12, color, 2)
        
        # 텍스트 정보
        if not np.isnan(zed_depth_mm):
            text_lines = [
                f"ID {marker_id}",
                f"Actual: {actual_mm/1000:.2f}m",
                f"ZED: {zed_depth_mm/1000:.2f}m",
                f"Error: {error_pct:.1f}%"
            ]
        else:
            text_lines = [
                f"ID {marker_id}",
                f"Actual: {actual_mm/1000:.2f}m",
                f"ZED: NaN"
            ]
        
        # 텍스트 배경 (가독성 향상)
        y_offset = py + 30
        for i, text in enumerate(text_lines):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 배경 사각형
            bg_y1 = y_offset + i*30 - 5
            bg_y2 = y_offset + i*30 + text_size[1] + 5
            bg_x1 = px + 20 - 5
            bg_x2 = px + 20 + text_size[0] + 5
            
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            
            # 텍스트
            cv2.putText(image, text, (px + 20, y_offset + i*30 + text_size[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def save_result(self, marker_id, px, py, zed_depth_mm, actual_mm):
        """
        결과 저장
        
        Args:
            marker_id: 마커 ID
            px, py: 픽셀 좌표
            zed_depth_mm: ZED depth (mm)
            actual_mm: 실제 거리 (mm)
        """
        if not np.isnan(zed_depth_mm) and not np.isinf(zed_depth_mm):
            error_mm = abs(zed_depth_mm - actual_mm)
            error_pct = (error_mm / actual_mm) * 100
            
            self.results.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'marker_id': marker_id,
                'pixel_x': px,
                'pixel_y': py,
                'actual_mm': actual_mm,
                'zed_mm': zed_depth_mm,
                'error_mm': error_mm,
                'error_pct': error_pct
            })
    
    def save_to_csv(self):
        """결과를 CSV로 저장"""
        if not self.results:
            print("⚠ 저장할 결과가 없습니다.")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f"aruco_validation_{timestamp}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"✓ CSV 저장: {csv_path}")
        return csv_path
    
    def print_summary(self):
        """결과 요약 출력"""
        if not self.results:
            print("⚠ 결과가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("결과 요약")
        print("="*60)
        
        # 마커별 통계
        marker_stats = {}
        for r in self.results:
            mid = r['marker_id']
            if mid not in marker_stats:
                marker_stats[mid] = []
            marker_stats[mid].append(r['error_pct'])
        
        print("\n[마커별 평균 오차]")
        for mid in sorted(marker_stats.keys()):
            errors = marker_stats[mid]
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            actual_m = self.marker_distances[mid]
            
            print(f"  ID {mid} ({actual_m}m): {avg_error:.2f}% ± {std_error:.2f}% (n={len(errors)})")
        
        # 전체 통계
        all_errors = [r['error_pct'] for r in self.results]
        print(f"\n[전체 통계]")
        print(f"  평균 오차: {np.mean(all_errors):.2f}%")
        print(f"  표준편차: {np.std(all_errors):.2f}%")
        print(f"  최소 오차: {np.min(all_errors):.2f}%")
        print(f"  최대 오차: {np.max(all_errors):.2f}%")
        print(f"  측정 횟수: {len(self.results)}회")
        
        print("="*60 + "\n")
    
    def run(self):
        """메인 실행 루프"""
        print("\n" + "="*60)
        print("ArUco Depth 검증 시작")
        print("="*60 + "\n")
        
        # ZED 초기화
        print("ZED 카메라 초기화 중...")
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"❌ ZED 카메라 열기 실패: {status}")
            return
        
        print("✓ ZED 카메라 초기화 완료\n")
        
        # 버퍼
        image = sl.Mat()
        depth = sl.Mat()
        
        print("="*60)
        print("키보드 조작:")
        print("  's' - 현재 프레임 저장")
        print("  'q' - 종료")
        print("="*60 + "\n")
        
        frame_count = 0
        
        try:
            while True:
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    # 이미지, depth 가져오기
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    
                    # numpy 변환
                    image_np = image.get_data()[:, :, :3]  # BGRA -> BGR
                    depth_np = depth.get_data()
                    
                    # ArUco 검출
                    markers = self.detect_aruco(image_np)
                    
                    # 시각화용 이미지
                    display = image_np.copy()
                    
                    # 각 마커 처리
                    for marker_id, (px, py, corners) in markers.items():
                        if marker_id in self.marker_distances:
                            # 5x5 윈도우 중앙값으로 depth 측정
                            zed_depth_mm = self.get_depth_median(depth_np, px, py, window_size=5)
                            actual_mm = self.marker_distances[marker_id] * 1000
                            
                            if not np.isnan(zed_depth_mm):
                                error_mm = abs(zed_depth_mm - actual_mm)
                                error_pct = (error_mm / actual_mm) * 100
                            else:
                                error_pct = np.nan
                            
                            # 시각화
                            self.draw_marker_info(
                                display, marker_id, px, py, corners,
                                zed_depth_mm, actual_mm, error_pct
                            )
                    
                    # 상태 정보 표시
                    info_text = f"Saved: {len(self.results)} | Press 's' to save, 'q' to quit"
                    cv2.putText(display, info_text, (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(display, info_text, (20, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                    
                    # 화면 표시
                    cv2.imshow("ArUco Depth Validation", display)
                    
                    # 키 입력 처리
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    
                    elif key == ord('s'):
                        # 현재 프레임의 모든 마커 저장
                        saved_count = 0
                        for marker_id, (px, py, corners) in markers.items():
                            if marker_id in self.marker_distances:
                                zed_depth_mm = self.get_depth_median(depth_np, px, py, window_size=5)
                                actual_mm = self.marker_distances[marker_id] * 1000
                                
                                self.save_result(marker_id, px, py, zed_depth_mm, actual_mm)
                                saved_count += 1
                        
                        if saved_count > 0:
                            print(f"✓ {saved_count}개 마커 저장됨 (총 {len(self.results)}개)")
                            
                            # 이미지도 저장
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            img_path = self.output_dir / f"capture_{timestamp}.jpg"
                            cv2.imwrite(str(img_path), display)
                        else:
                            print("⚠ 저장할 마커가 없습니다.")
                    
                    frame_count += 1
        
        except KeyboardInterrupt:
            print("\n\n⚠ 사용자에 의해 중단됨")
        
        finally:
            # 종료 처리
            zed.close()
            cv2.destroyAllWindows()
            
            # 결과 저장 및 요약
            if self.results:
                self.save_to_csv()
                self.print_summary()
            else:
                print("\n⚠ 저장된 결과가 없습니다.")


def main():
    # 설정 파일 확인
    if not Path('marker_config.yaml').exists():
        print("❌ marker_config.yaml 파일을 찾을 수 없습니다.")
        print("먼저 marker_config.yaml을 생성하세요.")
        return
    
    # 검증 실행
    validator = ArucoDepthValidator()
    validator.run()


if __name__ == "__main__":
    main()

