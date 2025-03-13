# main_3d.py
import cv2
import traceback
import numpy as np
import argparse
import os
import sys
from pose_estimator import PoseEstimator3D
from kalman_filter import KalmanFilterTracker3D
from skeleton_visualizer import SkeletonVisualizer
from com_calculator import COMCalculator

class PoseTrackingSystem3D:
    def __init__(self, roi_ratio=0.95):
        self.pose_estimator = PoseEstimator3D(roi_ratio=roi_ratio)
        self.kalman_tracker = KalmanFilterTracker3D()
        self.skeleton_visualizer = SkeletonVisualizer()
        self.com_calculator = COMCalculator()

        # CoM 이동 궤적 저장을 위한 버퍼
        self.com_trajectory = []
        self.max_trajectory_length = 60  # 2초 궤적 (30fps 기준)

    def process_frame(self, frame):
        """
        프레임 처리 및 3D 포즈 추적
        Args:
            frame (numpy.ndarray): 입력 프레임
        Returns:
            numpy.ndarray: 추적 결과가 표시된 프레임
        """
        try:
            landmarks, processed_frame = self.pose_estimator.estimate_pose(frame)

            if landmarks:
                # 3D 키포인트 추출
                keypoints_3d = self.pose_estimator.extract_3d_keypoints(landmarks, frame)

                # 칼만 필터로 키포인트 필터링
                filtered_keypoints_3d = self.kalman_tracker.track(keypoints_3d)

                # 3D CoM 계산
                com_3d = self.com_calculator.calculate_whole_body_com(filtered_keypoints_3d, include_z=True)

                if com_3d:
                    # CoM 궤적 업데이트
                    self.com_trajectory.append(com_3d)
                    if len(self.com_trajectory) > self.max_trajectory_length:
                        self.com_trajectory.pop(0)

                    # 현재 CoM 표시
                    cv2.circle(processed_frame, (int(com_3d[0]), int(com_3d[1])), 8, (0, 255, 255), -1)
                    cv2.putText(processed_frame, f"CoM z:{int(com_3d[2])}",
                                (int(com_3d[0]) + 10, int(com_3d[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # CoM 궤적 그리기
                    for i in range(1, len(self.com_trajectory)):
                        prev_com = self.com_trajectory[i-1]
                        curr_com = self.com_trajectory[i]

                        # 색상 그라데이션 (오래된 포인트일수록 더 투명하게)
                        alpha = i / len(self.com_trajectory)

                        # Z 값에 따른 색상 변화 (깊이 시각화)
                        z_val = curr_com[2]
                        z_color = (
                            0,
                            int(255 * alpha),
                            int(255 * (1.0 - abs(z_val) / 100) * alpha)  # Z값에 따른 색상 변화
                        )

                        cv2.line(processed_frame,
                                (int(prev_com[0]), int(prev_com[1])),
                                (int(curr_com[0]), int(curr_com[1])),
                                z_color, 2)

                    # 3D 이동 방향 계산 및 시각화
                    if len(self.com_trajectory) >= 5:
                        direction, speed = self.com_calculator.calculate_movement_direction(
                            self.com_trajectory, window_size=5)

                        # 방향 시각화
                        processed_frame = self.skeleton_visualizer.draw_direction_arrow(
                            processed_frame, com_3d, direction, speed)

                # 3D 스켈레톤 시각화
                processed_frame = self.skeleton_visualizer.draw_3d_skeleton(
                    processed_frame, keypoints_3d, filtered_keypoints_3d)

            return processed_frame

        except Exception as e:
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            return frame

    def run(self, input_source=0):
        """
        비디오 캡처 및 처리
        Args:
            input_source: 카메라 인덱스(정수) 또는 비디오 파일 경로(문자열)
        """
        try:
            # 창 이름 설정
            window_name = "Pose Tracking"

            # 입력 소스 설정
            if isinstance(input_source, str) and os.path.exists(input_source):
                # 비디오 파일 입력
                cap = cv2.VideoCapture(input_source)
                print(f"비디오 파일 열기: {input_source}")

                # 비디오 정보 출력
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"비디오 정보: {width}x{height}, {fps}fps, {frame_count}프레임")
            else:
                # 카메라 입력
                try:
                    # macOS에서는 AVFOUNDATION 백엔드 사용
                    cap = cv2.VideoCapture(input_source, cv2.CAP_AVFOUNDATION)
                except:
                    # 다른 플랫폼에서는 기본 백엔드 사용
                    cap = cv2.VideoCapture(input_source)

                # 카메라 속성 설정 (720p)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"카메라 열기: {input_source}")

            # 카메라/비디오 열기 확인
            if not cap.isOpened():
                print(f"Error: 입력 소스를 열 수 없습니다: {input_source}")
                return

            # 첫 프레임을 읽어서 창 크기 계산에 사용
            ret, first_frame = cap.read()
            if not ret:
                print("Error: 첫 프레임을 읽을 수 없습니다.")
                return

            # 첫 프레임의 크기 가져오기
            frame_height, frame_width = first_frame.shape[:2]

            # 기본 화면 해상도 기본값
            screen_width, screen_height = 1440, 900

            # 창 위치 계산
            x_position = int((screen_width - frame_width) / 2)
            y_position = int((screen_height - frame_height) / 2)

            # 첫 프레임 처리
            processed_first_frame = self.process_frame(first_frame)

            # 창 생성 및 위치 지정
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, processed_first_frame)
            cv2.moveWindow(window_name, x_position, y_position)

            # 프레임 카운터 초기화
            frame_count = 1

            # 출력 비디오 설정 (input이 비디오 파일인 경우)
            output_video = None
            if isinstance(input_source, str) and os.path.exists(input_source):
                output_path = f"output_{os.path.basename(input_source)}"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                print(f"출력 비디오 생성: {output_path}")

                # 첫 프레임 저장
                if output_video is not None:
                    output_video.write(processed_first_frame)

            while True:
                ret, frame = cap.read()

                if not ret:
                    if isinstance(input_source, str) and os.path.exists(input_source):
                        print(f"비디오 처리 완료: {frame_count} 프레임")
                    else:
                        print("Error: 프레임 캡처 실패")
                    break

                # 프레임 카운터 증가
                frame_count += 1

                # 프레임 처리
                processed_frame = self.process_frame(frame)

                # 처리된 프레임 표시 (창 위치는 이미 설정됨)
                cv2.imshow(window_name, processed_frame)

                # 출력 비디오에 프레임 추가 (비디오 파일 처리 시)
                if output_video is not None:
                    output_video.write(processed_frame)

                # 비디오 파일 처리 시 진행 상황 표시 (100프레임마다)
                if isinstance(input_source, str) and frame_count % 100 == 0:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"처리 중: {frame_count}/{total_frames} 프레임 ({frame_count/total_frames*100:.1f}%)")

                # ESC 키 또는 q로 종료
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()

        finally:
            cap.release()
            if output_video is not None:
                output_video.release()
            cv2.destroyAllWindows()

# 메인 함수 수정
if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Pose Tracking with CoM and Movement Direction')
    parser.add_argument('-i', '--input', type=str, default='0',
                        help='입력 소스 (카메라 인덱스 또는 비디오 파일 경로). 기본값: 0 (기본 카메라)')
    parser.add_argument('-r', '--roi', type=float, default=0.95,
                        help='ROI 비율 (0.0~1.0). 기본값: 0.95')
    args = parser.parse_args()

    # 입력 소스 처리
    if args.input.isdigit():
        input_source = int(args.input)
    else:
        input_source = args.input
        if not os.path.exists(input_source):
            print(f"Error: 입력 파일을 찾을 수 없습니다: {input_source}")
            sys.exit(1)

    # 시스템 초기화 및 실행
    pose_tracking = PoseTrackingSystem3D(roi_ratio=args.roi)
    pose_tracking.run(input_source)
