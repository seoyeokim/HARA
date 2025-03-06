# main.py
import cv2
import traceback
import numpy as np
from pose_estimator import PoseEstimator
from kalman_filter import KalmanFilterTracker
from skeleton_visualizer import SkeletonVisualizer
from com_calculator import COMCalculator

class PoseTrackingSystem:
    def __init__(self, roi_ratio=0.8):  # 짧은 차원의 80%를 ROI로 사용
        self.pose_estimator = PoseEstimator(roi_ratio=roi_ratio)
        self.kalman_tracker = KalmanFilterTracker()
        self.skeleton_visualizer = SkeletonVisualizer()
        self.com_calculator = COMCalculator()

        # CoM 이동 궤적 저장을 위한 버퍼
        self.com_trajectory = []
        self.max_trajectory_length = 60  # 2초 궤적 (30fps 기준)

    def process_frame(self, frame):
        """
        프레임 처리 및 포즈 추적
        Args:
            frame (numpy.ndarray): 입력 프레임
        Returns:
            numpy.ndarray: 추적 결과가 표시된 프레임
        """
        try:
            landmarks, processed_frame = self.pose_estimator.estimate_pose(frame)

            if landmarks:
                height, width, _ = frame.shape

                # 원본 키포인트 좌표 추출
                keypoints = [
                    (int(landmark.x * width), int(landmark.y * height))
                    for landmark in landmarks.landmark
                ]

                # 칼만 필터로 키포인트 필터링
                filtered_keypoints = self.kalman_tracker.track(keypoints)

                # CoM 계산
                com = self.com_calculator.calculate_whole_body_com(filtered_keypoints)

                if com:
                    # CoM 궤적 업데이트
                    self.com_trajectory.append(com)
                    if len(self.com_trajectory) > self.max_trajectory_length:
                        self.com_trajectory.pop(0)

                    # 현재 CoM 표시
                    cv2.circle(processed_frame, (int(com[0]), int(com[1])), 8, (0, 255, 255), -1)
                    cv2.putText(processed_frame, "CoM", (int(com[0]) + 10, int(com[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # CoM 궤적 그리기
                    for i in range(1, len(self.com_trajectory)):
                        prev_com = self.com_trajectory[i-1]
                        curr_com = self.com_trajectory[i]

                        # 색상 그라데이션 (오래된 포인트일수록 더 투명하게)
                        alpha = i / len(self.com_trajectory)
                        color = (0, int(255 * alpha), int(255 * alpha))

                        cv2.line(processed_frame,
                                (int(prev_com[0]), int(prev_com[1])),
                                (int(curr_com[0]), int(curr_com[1])),
                                color, 2)

                # 스켈레톤 시각화
                processed_frame = self.skeleton_visualizer.draw_2d_skeleton(
                    processed_frame, landmarks, filtered_keypoints)

            return processed_frame

        except Exception as e:
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            return frame

    def run(self):
        """비디오 캡처 및 처리"""
        try:
            # macOS에서 안정적인 카메라 캡처를 위한 추가 설정
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

            # 카메라 속성 설정 (720p)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # 카메라 연결 확인
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to capture frame")
                    break

                processed_frame = self.process_frame(frame)

                cv2.imshow("Pose Estimation with CoM", processed_frame)

                # ESC 키 또는 q로 종료
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()

        finally:
            cap.release()
            cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    pose_tracking = PoseTrackingSystem(roi_ratio=0.8)
    pose_tracking.run()
