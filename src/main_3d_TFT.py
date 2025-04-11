# main_3d.py 업데이트된 버전 (이진 분류 TFT 모델 지원)
import cv2
import numpy as np
import argparse
import os
import sys
import platform
import logging
from pose_estimator import PoseEstimator3D
from kalman_filter import KalmanFilterTracker3D
from skeleton_visualizer import SkeletonVisualizer
from com_calculator import COMCalculator

# TFT 모듈 조건부 임포트 부분
try:
    from TFT_inference import TFTInferenceSystem
    TFT_AVAILABLE = True
    # 모듈을 임포트할 수 있지만 binary_mode 파라미터가 있는지 확인
    import inspect
    tft_params = inspect.signature(TFTInferenceSystem.__init__).parameters
    BINARY_MODE_SUPPORTED = 'binary_mode' in tft_params
    if not BINARY_MODE_SUPPORTED:
        print("경고: TFT_inference.py 파일이 binary_mode를 지원하지 않습니다. binary_mode는 무시됩니다.")
except ImportError:
    TFT_AVAILABLE = False
    BINARY_MODE_SUPPORTED = False
    print("TFT 추론 모듈을 찾을 수 없습니다. TFT 추론 기능이 비활성화됩니다.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PoseTrackingSystem3D:
    def __init__(self, roi_padding, roi_ratio, use_tft=False, tft_checkpoint_dir=None,
                tft_sequence_length=20, tft_binary_mode=False, csv_export=True,
                csv_filename='keypoints_export_normalized.csv'):
        """
        3D 포즈 추적 시스템 초기화
        Args:
            roi_padding (int): 감지된 랜드마크 주변 ROI 여유 공간 (픽셀)
            roi_ratio (float): ROI 크기 비율 (0.0 ~ 1.0)
            use_tft (bool): TFT 추론 기능 사용 여부
            tft_checkpoint_dir (str): TFT 모델 체크포인트 디렉터리 경로
            tft_sequence_length (int): TFT 모델 입력 시퀀스 길이
            tft_binary_mode (bool): 2클래스 TFT 모델 사용 여부 (True: 바이너리 모드, False: 4클래스 모드)
            csv_export (bool): 키포인트 데이터를 CSV로 내보낼지 여부
            csv_filename (str): 내보낼 CSV 파일 이름 (기본값 변경: 정규화된 데이터임을 명시)
        """
        self.pose_estimator = PoseEstimator3D(roi_padding=roi_padding, roi_ratio=roi_ratio)
        self.kalman_tracker = KalmanFilterTracker3D()
        self.skeleton_visualizer = SkeletonVisualizer()
        self.com_calculator = COMCalculator()

        # CoM 이동 궤적 저장을 위한 버퍼
        self.com_trajectory = []
        self.max_trajectory_length = 60  # 2초 궤적 (30fps 기준)

        # 안정화를 위한 변수
        self.stable_com_pos = None
        self.max_allowed_move = 30.0  # 한 프레임당 최대 허용 이동 거리
        self.smoothing_alpha = 0.7    # 지수 이동 평균 알파 값

        # Z값과 Z방향 구분 시각화를 위한 설정
        self.z_color_scale = 5.0  # Z값 색상 변화 스케일 계수
        self.z_direction_threshold = 0.3  # 의미 있는 Z방향 움직임 임계값

        # Z축 방향 표시
        self.z_direction_history = {"direction": None, "strength": 0, "duration": 0}
        self.z_direction_min_duration = 30  # 최소 표시 지속 프레임 수 (약 1초로 수정)
        self.z_direction_max_duration = 60  # 최대 표시 지속 프레임 수 (약 2초로 수정)

        # TFT 바이너리 모드 설정
        self.tft_binary_mode = tft_binary_mode

        # CSV 내보내기 설정
        self.csv_export = csv_export
        self.csv_filename = csv_filename

        # TFT 추론 시스템 초기화
        self.use_tft = use_tft and TFT_AVAILABLE
        self.tft_system = None

        if self.use_tft:
            if tft_checkpoint_dir is None or not os.path.exists(tft_checkpoint_dir):
                logger.warning("유효한 TFT 체크포인트 디렉터리가 제공되지 않았습니다. TFT 추론이 비활성화됩니다.")
                self.use_tft = False
            else:
                try:
                    # binary_mode 파라미터 지원 여부에 따라 다르게 초기화
                    if BINARY_MODE_SUPPORTED:
                        self.tft_system = TFTInferenceSystem(
                            checkpoint_dir=tft_checkpoint_dir,
                            max_encoder_length=tft_sequence_length,
                            use_mps=True,
                            binary_mode=tft_binary_mode,
                            csv_export=csv_export,
                            csv_filename=csv_filename
                        )
                    else:
                        # binary_mode를 지원하지 않는 경우
                        self.tft_system = TFTInferenceSystem(
                            checkpoint_dir=tft_checkpoint_dir,
                            max_encoder_length=tft_sequence_length,
                            use_mps=True,
                            csv_export=csv_export,
                            csv_filename=csv_filename
                        )
                    logger.info(f"TFT 추론 시스템이 성공적으로 초기화되었습니다. 클래스 수: {'2' if tft_binary_mode else '4'}")
                    if csv_export:
                        logger.info(f"키포인트 데이터를 {csv_filename} 파일로 내보냅니다.")
                except Exception as e:
                    logger.error(f"TFT 추론 시스템 초기화 실패: {e}")
                    self.use_tft = False

    def _calculate_com(self, filtered_keypoints_3d):
        """
        무게중심 계산 및 안정화
        Args:
            filtered_keypoints_3d: 필터링된 3D 키포인트
        Returns:
            tuple: 안정화된 CoM 위치 (x, y, z)
        """
        # 3D CoM 계산
        com_3d = self.com_calculator.calculate_whole_body_com(filtered_keypoints_3d, include_z=True)
        if not com_3d:
            return None

        # CoM 필터링 - 반응성 개선
        if self.stable_com_pos is None:
            # 첫 프레임은 그대로 사용
            self.stable_com_pos = com_3d
        else:
            # 이전 CoM 위치와 현재 위치 사이의 거리 계산
            prev_com = np.array(self.stable_com_pos)
            current_com = np.array(com_3d)
            distance = np.linalg.norm(current_com - prev_com)

            # 속도에 따른 적응형 알파값 (더 빠른 움직임에는 더 높은 알파값)
            adaptive_alpha = min(0.9, self.smoothing_alpha + (distance / 100.0))

            # 급격한 변화 필터링 (거리가 큰 경우 부드럽게 보간)
            if distance > self.max_allowed_move:
                # 최대 거리로 제한된 새 위치 계산
                direction = (current_com - prev_com) / distance
                limited_pos = prev_com + direction * self.max_allowed_move
                com_3d = tuple(limited_pos)
            else:
                # 지수 이동 평균 적용
                smoothed_com = prev_com * (1 - adaptive_alpha) + current_com * adaptive_alpha
                com_3d = tuple(smoothed_com)

            self.stable_com_pos = com_3d

        # CoM 궤적 업데이트
        self.com_trajectory.append(com_3d)
        if len(self.com_trajectory) > self.max_trajectory_length:
            self.com_trajectory.pop(0)

        return com_3d

    def _visualize_com(self, frame, com_3d):
        """
        CoM 시각화 - Z값(절대적 깊이)을 색상으로 표현
        Args:
            frame: 처리할 프레임
            com_3d: CoM 위치 (x, y, z)
        Returns:
            numpy.ndarray: CoM이 시각화된 프레임
        """
        # Z값에 따른 색상 매핑
        z_value = com_3d[2]

        # Z값 범위를 색상으로 변환 (그라데이션)
        # 기준점(z=0)에서는 녹색, 양수(멀어짐)는 녹색→노란색, 음수(가까워짐)는 녹색→청록색
        if z_value > 0:  # 기준보다 멀리 있음
            # 녹색에서 노란색으로 그라데이션 (0,255,0) -> (0,255,255) [BGR 순서]
            intensity = min(255, int(abs(z_value) * 10))
            depth_color = (0, 255, intensity)
        else:  # 기준보다 가까이 있음
            # 녹색에서 청록색으로 그라데이션 (0,255,0) -> (255,255,0) [BGR 순서]
            intensity = min(255, int(abs(z_value) * 10))
            depth_color = (intensity, 255, 0)

        # 현재 CoM 표시 (색상은 절대적 Z값 기준)
        cv2.circle(frame, (int(com_3d[0]), int(com_3d[1])), 8, depth_color, -1)

        # Z값 텍스트 표시
        cv2.putText(frame, f"CoM z:{int(z_value)}",
                (int(com_3d[0]) + 10, int(com_3d[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def _draw_trajectory(self, frame, com_3d):
        """
        CoM 궤적 그리기
        Args:
            frame: 처리할 프레임
            com_3d: 현재 CoM 위치
        Returns:
            numpy.ndarray: 궤적이 그려진 프레임
        """
        # 궤적이 충분히 쌓이지 않았으면 건너뜀
        if len(self.com_trajectory) < 2:
            return frame

        # CoM 궤적 그리기
        for i in range(1, len(self.com_trajectory)):
            prev_com = self.com_trajectory[i-1]
            curr_com = self.com_trajectory[i]

            # 연속된 점 사이 거리 계산
            point_distance = np.linalg.norm(np.array(curr_com) - np.array(prev_com))

            # 거리가 너무 크면 선을 그리지 않음 (궤적 끊김 방지)
            if point_distance < self.max_allowed_move * 2:
                # 색상 그라데이션 (오래된 포인트일수록 더 투명하게)
                alpha = i / len(self.com_trajectory)

                # Z 값에 따른 색상 변화 (깊이 시각화)
                z_val = curr_com[2]
                z_color = (
                    0,
                    int(255 * alpha),
                    int(255 * (1.0 - abs(z_val) / 100) * alpha)  # Z값에 따른 색상 변화
                )

                cv2.line(frame,
                        (int(prev_com[0]), int(prev_com[1])),
                        (int(curr_com[0]), int(curr_com[1])),
                        z_color, 2)

        return frame

    def _draw_direction(self, frame, com_3d):
        """
        이동 방향 화살표 그리기
        Args:
            frame: 처리할 프레임
            com_3d: CoM 위치 (x, y, z)
        Returns:
            numpy.ndarray: 방향 화살표가 그려진 프레임
        """
        # 궤적이 충분히 쌓이지 않았으면 건너뜀
        if len(self.com_trajectory) < 5:
            return frame

        try:
            # 이동 방향 계산
            direction_vector, speed = self.com_calculator.calculate_movement_direction(
                self.com_trajectory, window_size=5)

            # 2D 방향 화살표는 항상 그리기
            frame = self.skeleton_visualizer.draw_direction_arrow(
                frame, com_3d, direction_vector[:2] if len(direction_vector) == 3 else direction_vector, speed)

            # # Z축 방향 처리 조건 확인
            # if (len(direction_vector) == 3 and
            #     hasattr(self.pose_estimator, 'z_baseline') and
            #     self.pose_estimator.z_baseline is not None):

            #     z_direction = direction_vector[2]
            #     z_threshold = 0.9

            #     # 의미 있는 Z 방향 감지 시
            #     if abs(z_direction) > z_threshold:
            #         # 현재 방향과 강도 저장
            #         self.z_direction_history["direction"] = z_direction
            #         self.z_direction_history["strength"] = abs(z_direction)

            #         # 간단한 지속 시간 설정: 기본 30프레임, 강한 움직임이면 최대 60프레임
            #         base_duration = 30  # 기본 지속 시간 (약 1초)
            #         extra_duration = int(min(30, abs(z_direction) * 50))  # 강도에 비례한 추가 지속 시간
            #         self.z_direction_history["duration"] = base_duration + extra_duration

            #     # Z 방향 화살표 표시 (지속 시간이 남아있는 경우)
            #     if self.z_direction_history["duration"] > 0:
            #         # Z 방향 벡터 생성 (현재 XY 방향과 저장된 Z 방향 결합)
            #         z_arrow_direction = (
            #             direction_vector[0],
            #             direction_vector[1],
            #             self.z_direction_history["direction"]
            #         )

            #         # Z 방향 시각화
            #         frame = self.skeleton_visualizer.draw_3d_direction(
            #             frame, com_3d, z_arrow_direction, speed)

            #         # 매 프레임마다 지속 시간 감소
            #         self.z_direction_history["duration"] -= 1

            #         # 지속 시간이 끝났을 때 초기화
            #         if self.z_direction_history["duration"] <= 0:
            #             self.z_direction_history["direction"] = None
            #             self.z_direction_history["strength"] = 0

        except Exception as e:
            logger.error(f"Error drawing direction arrow: {e}", exc_info=True)

        return frame

    def process_frame(self, frame):
        """
        프레임 처리 및 3D 포즈 추적 - Z값과 Z방향 구분 시각화
        Args:
            frame (numpy.ndarray): 입력 프레임
        Returns:
            numpy.ndarray: 추적 결과가 표시된 프레임
        """
        processed_frame = frame.copy()

        try:
            # 1. 포즈 추정
            landmarks, processed_frame = self.pose_estimator.estimate_pose(frame)
            if not landmarks:
                return processed_frame

            # 2. 키포인트 추출 및 필터링
            keypoints_3d = self.pose_estimator.extract_3d_keypoints(landmarks, frame)
            filtered_keypoints_3d = self.kalman_tracker.track(keypoints_3d)

            # 3. CoM 계산 및 안정화
            com_3d = self._calculate_com(filtered_keypoints_3d)
            if not com_3d:
                return processed_frame

            # 4. TFT 행동 예측 처리 (활성화된 경우)
            if self.use_tft and self.tft_system:
                predicted_class, predicted_class_name, probabilities, prediction_status = self.tft_system.process_keypoints(
                    filtered_keypoints_3d, com_3d)

                # 예측 결과 시각화
                processed_frame = self.tft_system.visualize_prediction(
                    processed_frame, predicted_class, predicted_class_name, probabilities, prediction_status)

            # 5. 시각화
            # 5.1 CoM 시각화 (Z값 기준 색상)
            processed_frame = self._visualize_com(processed_frame, com_3d)

            # 5.2 궤적 그리기
            processed_frame = self._draw_trajectory(processed_frame, com_3d)

            # 5.3 이동 방향 화살표 그리기 (Z방향 별도 표시)
            processed_frame = self._draw_direction(processed_frame, com_3d)

            # 5.4 스켈레톤 시각화
            processed_frame = self.skeleton_visualizer.draw_3d_skeleton(
                processed_frame, keypoints_3d, filtered_keypoints_3d)

            # 5.5 컨트롤 안내 추가 (화면 하단)
            frame_height, frame_width = processed_frame.shape[:2]
            cv2.putText(processed_frame, "ESC or Q: Exit",
                    (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # TFT 상태 표시 (활성화된 경우)
            if self.use_tft:
                mode_text = "Binary" if self.tft_binary_mode else "4-Class"
                cv2.putText(processed_frame, f"TFT {mode_text}: Active",
                        (frame_width - 200, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            # 오류 발생 시 원본 프레임 반환
            return frame

        return processed_frame

    def run(self, input_source=0):
        """
        비디오 캡처 및 처리
        Args:
            input_source: 카메라 인덱스(정수) 또는 비디오 파일 경로(문자열)
        """
        cap = None
        output_video = None

        try:
            # 1. 입력 소스 설정
            cap = self._setup_capture(input_source)
            if not cap or not cap.isOpened():
                logger.error(f"Failed to open input source: {input_source}")
                return

            # 2. 첫 프레임 처리 및 창 설정
            ret, first_frame = cap.read()
            if not ret:
                logger.error("Failed to read first frame")
                return

            processed_first_frame = self.process_frame(first_frame)
            window_name = "Pose Tracking"

            # 3. 창 위치 설정
            frame_height, frame_width = first_frame.shape[:2]
            screen_width, screen_height = 1440, 900  # 기본 화면 해상도
            x_position = int((screen_width - frame_width) / 2)
            y_position = int((screen_height - frame_height) / 2)

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, processed_first_frame)
            cv2.moveWindow(window_name, x_position, y_position)

            # 4. 출력 비디오 설정 (비디오 파일 처리 시)
            if isinstance(input_source, str) and os.path.exists(input_source):
                output_video = self._setup_output_video(input_source, cap)
                if output_video:
                    output_video.write(processed_first_frame)

            # 5. 메인 처리 루프
            frame_count = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    self._handle_end_of_video(input_source, frame_count)
                    break

                # 프레임 카운터 증가
                frame_count += 1

                # 프레임 처리
                processed_frame = self.process_frame(frame)
                cv2.imshow(window_name, processed_frame)

                # 출력 비디오에 프레임 추가 (비디오 파일 처리 시)
                if output_video is not None:
                    output_video.write(processed_frame)

                # 비디오 파일 처리 시 진행 상황 표시 (100프레임마다)
                if isinstance(input_source, str) and frame_count % 100 == 0:
                    self._show_progress(cap, frame_count)

                # ESC 키 또는 q로 종료
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)

        finally:
            # 리소스 해제
            if cap:
                cap.release()
            if output_video:
                output_video.release()
            cv2.destroyAllWindows()

    def _setup_capture(self, input_source):
        """
        비디오 캡처 설정
        Args:
            input_source: 카메라 인덱스 또는 비디오 파일 경로
        Returns:
            cv2.VideoCapture: 설정된 캡처 객체
        """
        os_name = platform.system()

        if isinstance(input_source, str) and os.path.exists(input_source):
            # 비디오 파일 입력
            cap = cv2.VideoCapture(input_source)
            logger.info(f"Opening video file: {input_source}")

            # 비디오 정보 출력
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Video info: {width}x{height}, {fps}fps, {frame_count} frames")
        else:
            # 카메라 입력
            logger.info(f"Attempting to open camera: {input_source} on {os_name}")

            if os_name == "Darwin": # macOS
                # macOS에서는 AVFOUNDATION 백엔드 사용
                cap = cv2.VideoCapture(input_source, cv2.CAP_AVFOUNDATION)
            elif os_name == "Windows":
                # Windows에서는 DSHOW 백엔드 사용
                cap = cv2.VideoCapture(input_source, cv2.CAP_DSHOW)
            else: # Linux 등 기타
                cap = cv2.VideoCapture(input_source, cv2.CAP_V4L2)

            # 카메라 속성 설정 (720p)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            logger.info(f"Opening camera: {input_source}")

        return cap

    def _setup_output_video(self, input_source, cap):
        """
        출력 비디오 설정
        Args:
            input_source: 입력 비디오 파일 경로
            cap: 입력 비디오 캡처 객체
        Returns:
            cv2.VideoWriter: 설정된 비디오 라이터 객체
        """
        output_path = f"output_{os.path.basename(input_source)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'avc1', 'H264'
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Creating output video: {output_path}")

        return output_video

    def _handle_end_of_video(self, input_source, frame_count):
        """
        비디오 종료 처리
        Args:
            input_source: 입력 소스
            frame_count: 처리된 프레임 수
        """
        if isinstance(input_source, str) and os.path.exists(input_source):
            logger.info(f"Video processing completed: {frame_count} frames")
        else:
            logger.error("Frame capture failed")

    def _show_progress(self, cap, frame_count):
        """
        처리 진행 상황 표시
        Args:
            cap: 비디오 캡처 객체
            frame_count: 현재 프레임 수
        """
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        percentage = frame_count / total_frames * 100
        logger.info(f"Processing: {frame_count}/{total_frames} frames ({percentage:.1f}%)")

# main 함수에 인자 추가
def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Pose Tracking with CoM and Movement Direction')
    parser.add_argument('-i', '--input', type=str, default='0',
                        help='입력 소스 (카메라 인덱스 또는 비디오 파일 경로). 기본값: 0 (기본 카메라)')
    parser.add_argument('-p', '--padding', type=float, default=80,
                        help='ROI 여유 공간 (픽셀). 기본값: 80')
    parser.add_argument('-r', '--roi', type=float, default=0.95,
                        help='ROI 비율 (0.0~1.0). 기본값: 0.95')

    # TFT 관련 인자 추가
    parser.add_argument('--tft', action='store_true',
                        help='TFT 추론 기능 활성화 (기본값: 비활성화)')
    parser.add_argument('--tft-checkpoint', type=str,
                        help='TFT 모델 체크포인트 디렉터리 경로 (--tft가 활성화된 경우에만 사용)')
    parser.add_argument('--tft-sequence', type=int, default=20,
                        help='TFT 모델 입력 시퀀스 길이 (프레임 수). 기본값: 20')
    # 바이너리 모드 추가
    parser.add_argument('--binary', action='store_true',
                        help='2클래스 TFT 모델 사용 (Standing/Walking만 구분). 기본값: False (4클래스 모드)')

    # CSV 내보내기
    parser.add_argument('--csv-export', action='store_true', default=True,
                        help='키포인트 데이터를 CSV로 내보내기 (기본값: True)')
    parser.add_argument('--csv-filename', type=str, default='keypoints_export_normalized.csv',
                        help='내보낼 CSV 파일 이름 (기본값: keypoints_export_normalized.csv)')

    args = parser.parse_args()

    args = parser.parse_args()

    # 입력 소스 처리
    if args.input.isdigit():
        input_source = int(args.input)
    else:
        input_source = args.input
        if not os.path.exists(input_source):
            logger.error(f"Error: 입력 파일을 찾을 수 없습니다: {input_source}")
            sys.exit(1)

    # TFT 기능 활성화 여부 확인 및 경고
    if args.tft and not TFT_AVAILABLE:
        logger.warning("TFT 모듈을 임포트할 수 없어 TFT 추론 기능이 비활성화됩니다.")
        args.tft = False

    # 모델 체크포인트 경로 설정 (바이너리 모드에 따라 다른 기본 경로 사용)
    if args.tft and args.tft_checkpoint is None:
        if args.binary:
            args.tft_checkpoint = '../checkpoints/TFT/2class'
            logger.info(f"바이너리 모드용 기본 체크포인트 설정: {args.tft_checkpoint}")
        else:
            args.tft_checkpoint = '../checkpoints/TFT/4class'
            logger.info(f"4클래스 모드용 기본 체크포인트 설정: {args.tft_checkpoint}")

    # 추가 TFT 옵션 로깅
    if args.tft:
        mode_str = "이진 분류 (Standing/Walking)" if args.binary else "4클래스 분류"
        logger.info(f"TFT 추론 활성화: 체크포인트 = {args.tft_checkpoint}, 시퀀스 길이 = {args.tft_sequence}, 모드 = {mode_str}")
    else:
        logger.info("TFT 추론 비활성화")

    # CSV 내보내기 설정 로깅
    if args.csv_export:
        logger.info(f"CSV 내보내기 활성화: 파일명 = {args.csv_filename}")
    else:
        logger.info("CSV 내보내기 비활성화")

    # 시스템 초기화 및 실행
    pose_tracking = PoseTrackingSystem3D(
        roi_padding=args.padding,
        roi_ratio=args.roi,
        use_tft=args.tft,
        tft_checkpoint_dir=args.tft_checkpoint if args.tft else None,
        tft_sequence_length=args.tft_sequence,
        tft_binary_mode=args.binary,
        csv_export=args.csv_export,
        csv_filename=args.csv_filename
    )
    pose_tracking.run(input_source)

if __name__ == "__main__":
    main()
