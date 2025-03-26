import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import cv2
import glob
import argparse
import traceback
import numpy as np
import pandas as pd
from pose_estimator import PoseEstimator3D
from kalman_filter import KalmanFilterTracker3D
from kalman_filter import KeypointPreprocess
from com_calculator import COMCalculator

class DatasetCreater:
    def __init__(self, roi_ratio=0.8, data_path=None, save_path=None, class_type='nomal', point_remain='all'):
        self.pose_estimator = PoseEstimator3D(roi_ratio=roi_ratio)
        self.kalman_tracker = KalmanFilterTracker3D()
        self.com_calculator = COMCalculator()
        self.keypoint_preprocessing = KeypointPreprocess(True)
        
        self.files = []
        Cut_files = glob.glob(os.path.join(data_path+'Text', "*.txt"))
        for Cut_file in Cut_files:
            dataname = os.path.splitext(os.path.basename(Cut_file))[0]
            Video_file = os.path.join(data_path+'Video', dataname+'.mp4')
            self.files.append((Cut_file, Video_file))
        
        self.save_path = save_path
        self.class_type = class_type
        self.point_remain = point_remain
        
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
        
    def process_frame(self, frame):
        """
        프레임 처리 및 3D 포즈 추적
        Args:
            frame (numpy.ndarray): 입력 프레임
        Returns:
            numpy.ndarray: 추적 결과가 표시된 프레임
        """
        sequence_data = []
        
        try:
            landmarks, _ = self.pose_estimator.estimate_pose(frame)
            if landmarks:
                # 3D 키포인트 추출
                keypoints_3d = self.pose_estimator.extract_3d_keypoints(landmarks, frame)
                
                # 칼만 필터로 키포인트 필터링
                filtered_keypoints_3d = self.kalman_tracker.track(keypoints_3d)
                
                # 3D CoM 계산
                com_3d = self.com_calculator.calculate_whole_body_com(filtered_keypoints_3d, include_z=True)

                if self.point_remain == 'all':
                    keypoints_and_com = filtered_keypoints_3d[11:]
                elif self.point_remain == 'lower_only':
                    remain_points = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                    keypoints_and_com = [filtered_keypoints_3d[point] for point in remain_points]
                
                keypoints_and_com.append(com_3d)
                
                keypoints_and_com = self.keypoint_preprocessing.process(keypoints_and_com)
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
                
        if self.class_type == 'binary' :
            if label == 2 : label = 1
            if label == 3 : label = 1
        
        if self.class_type == 'masking' :
            label += 1
            
        return label
        
    def run(self):
        """
        비디오 캡처 및 처리
        
        MP4 파일을 프레임 수에 따라 라벨링 후 병합
        """
        for datas in self.files:
            (cut_file, video_path) = datas
            
            cap = cv2.VideoCapture(video_path)
            cuts = self.read_cut_file(cut_file)
            sequence_tensor = []
            
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
            csv_filename = os.path.splitext(os.path.basename(cut_file))[0] + "_dataset.csv"
            csv_file_path = os.path.join(self.save_path, csv_filename)
            df = pd.DataFrame(sequence_tensor)
            df.to_csv(csv_file_path, index=False, header=False)
            print(f"Dataset saved as {csv_filename}")

# 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D Pose Tracking on a video file.")
    parser.add_argument("load_path", type=str, help="Path to the load data.")
    parser.add_argument("save_path", type=str, help="Path to the save path.")
    parser.add_argument("class_type", type=str, help="Path to the class type.")
    parser.add_argument("point_remain", type=str, help="Path to the point remain.")
    args = parser.parse_args()
    
    pose_tracking = DatasetCreater(roi_ratio=0.8, data_path=args.load_path, save_path=args.save_path, class_type=args.class_type, point_remain=args.point_remain)
    pose_tracking.run()