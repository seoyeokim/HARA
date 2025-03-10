# main_3d.py
import cv2
import argparse
import traceback
import numpy as np
import pandas as pd
from pose_estimator import PoseEstimator3D
from kalman_filter import KalmanFilterTracker3D
from com_calculator import COMCalculator

class DatasetCreater:
    def __init__(self, roi_ratio=0.8, video_path=None):
        self.pose_estimator = PoseEstimator3D(roi_ratio=roi_ratio)
        self.kalman_tracker = KalmanFilterTracker3D()
        self.com_calculator = COMCalculator()
        self.video_path = video_path

    def coodinate_convert(self, keypoints):
        converted_keypoints = []
        
        for i in range(len(keypoints)):
            keypoints[i] = np.array(keypoints[i])
        
        [x1, x2] = [keypoints[13], keypoints[12]]
        [y1, y2] = [keypoints[1], keypoints[0]]
        
        s_vector = (y1 - x1) + (y2 - x2)
        
        x_vector = (x2 - x1)/np.linalg.norm(x2 - x1)
        z_vector = np.cross(x_vector, s_vector)
        z_vector /= np.linalg.norm(z_vector)
        
        y_vector = np.cross(x_vector, z_vector)
        y_vector /= np.linalg.norm(y_vector)
        
        rotation_matrix = np.column_stack((x_vector, y_vector, z_vector))
        middle_point = ((x1 + x2)/2) @ rotation_matrix
        
        for i in range(len(keypoints)):
            converted_point = tuple((keypoints[i] @ rotation_matrix) - middle_point)
            converted_keypoints.append(converted_point)
            
        return converted_keypoints
        
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
                
                keypoints_and_com = filtered_keypoints_3d[11:] 
                keypoints_and_com.append(com_3d)

                keypoints_and_com = self.coodinate_convert(keypoints_and_com)
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
        """비디오 캡처 및 처리"""
        try:
            # 비디오 파일을 사용하여 캡처
            if not self.video_path:
                print("Error: No video source provided.")
                return
            
            cap = cv2.VideoCapture(self.video_path)
            sequence_tensor = []
            
            if not cap.isOpened():
                print("Error: Could not open video source.")
                return
            
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("End of video or failed to capture frame")
                    break

                sequence_vector = self.process_frame(frame)
                sequence_vector.append(self.make_label(self.video_path))
                sequence_tensor.append(sequence_vector)
                
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

            csv_filename = self.video_path.rsplit('.')[0] + "_dataset.csv"
            df = pd.DataFrame(sequence_tensor)
            df.to_csv(csv_filename, index=False, header=False)
            print(f"Dataset saved as {csv_filename}")
            
# 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D Pose Tracking on a video file.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    args = parser.parse_args()
    
    pose_tracking = DatasetCreater(video_path=args.video_path)
    pose_tracking.run()