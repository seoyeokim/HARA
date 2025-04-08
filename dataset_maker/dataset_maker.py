import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import cv2
import argparse
import traceback
import numpy as np
import pandas as pd
from pose_estimator import PoseEstimator3D
from kalman_filter import KalmanFilterTracker3D
from kalman_filter import KeypointPreprocess
from com_calculator import COMCalculator

class DatasetCreater:
    def __init__(self, roi_ratio=0.8, roi_padding=80, video_path=None, cut_file=None):
        self.pose_estimator = PoseEstimator3D(roi_ratio=roi_ratio, roi_padding=roi_padding)
        self.kalman_tracker = KalmanFilterTracker3D()
        self.com_calculator = COMCalculator()
        self.keypoint_preprosessing = KeypointPreprocess(True)

        self.video_path = video_path
        self.cut_file = cut_file

        # 안정화를 위한 변수
        self.stable_com_pos = None
        self.max_allowed_move = 30.0  # 한 프레임당 최대 허용 이동 거리
        self.smoothing_alpha = 0.7    # 지수 이동 평균 알파 값

    def read_cut_file(self, cut_file):
        """
        cut.txt 파일을 읽어 (프레임 수, 파일 이름) 리스트 반환
        """
        cuts = []
        with open(cut_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    frame_count, label = parts
                    cuts.append((int(frame_count.strip()), label.strip()))
        return cuts

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

        return com_3d

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
                com_3d = self._calculate_com(filtered_keypoints_3d)

                keypoints_and_com = filtered_keypoints_3d[11:]
                keypoints_and_com.append(com_3d)

                keypoints_and_com = self.keypoint_preprosessing.process(keypoints_and_com)
                sequence_data = [x for y in keypoints_and_com for x in y]

            return sequence_data

        except Exception as e:
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            return frame

    def make_label(self, video_path):
        label_list = ['stand', 'forte', 'walk', 'piano']

        for labels in label_list:
            if labels in video_path:
                label = label_list.index(labels)

        return label

    def run(self):
        """
        비디오 캡처 및 처리

        MP4 파일을 프레임 수에 따라 라벨링 후 병합
        """
        # 비디오 파일을 사용하여 캡처
        if not self.video_path:
            print("Error: No video source provided.")
            return

        cap = cv2.VideoCapture(self.video_path)
        cuts = self.read_cut_file(self.cut_file)
        sequence_tensor, current_frame = [[] , 0]

        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        for frame_count, label in cuts:
            for _ in range(frame_count):
                ret, frame = cap.read()

                if not ret:
                    print("End of video or failed to capture frame")
                    break

                sequence_vector = self.process_frame(frame)
                sequence_vector.append(self.make_label(label))
                sequence_tensor.append(sequence_vector)

        cap.release()
        csv_filename = self.video_path.rsplit('.')[0] + "_dataset.csv"
        df = pd.DataFrame(sequence_tensor)
        df.to_csv(csv_filename, index=False, header=False)
        print(f"Dataset saved as {csv_filename}")


# 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D Pose Tracking on a video file.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("cut_file", type=str, help="Path to the cut.txt file")
    args = parser.parse_args()

    pose_tracking = DatasetCreater(roi_ratio=1, video_path=args.video_path, cut_file=args.cut_file)
    pose_tracking.run()
