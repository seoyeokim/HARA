import cv2
import mediapipe as mp
import numpy as np
from src.roi_tracker import Tracker, yoloTracker

class PoseEstimator3D:
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 tracker_type=None,  # or "mmtrack" or None
                 roi_padding=40,
                 roi_ratio=0.8):
        
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.roi_bbox = None
        self.former_roi = None
        self.roi_active = False
        self.roi_padding = roi_padding
        self.roi_ratio = roi_ratio

        # Z 안정화 관련
        self.z_history = {}
        self.z_history_max_length = 10
        self.z_baseline = None
        self.z_scale_factor = 0.5
        self.initialization_frames = 30
        self.frame_count = 0
        self.initial_z_values = []
        
        # 트래커 선택
        self.tracker_type = tracker_type
        self.tracker = None
        self.target_id = None

        if tracker_type: self.tracker = yoloTracker(roi_padding, roi_ratio, 0.1, 0.1)
        else: self.tracker = Tracker(roi_padding, roi_ratio)
            
    def estimate_pose(self, frame, frame_num=None, launch_sign=False):
        height, width = frame.shape[:2]
        output_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = frame_rgb
        new_roi = None
        
        # ROI 계산
        if self.tracker_type and self.tracker:
            try:
                if (frame_num % 3) == 0:
                    new_roi, self.target_id = self.tracker.computed_roi(frame, launch_sign, target_id=self.target_id)
                
                if new_roi:
                    self.roi_bbox = self.former_roi = new_roi
                    self.roi_active = True
            except Exception as e:
                print(f"[WARN] mmTracker ROI 실패: {e}")
                self.roi_active = False
                
        # ROI 내 영역 잘라서 처리
        offset_x, offset_y = 0, 0
        
        if self.roi_bbox:
            if self.roi_active and self.roi_bbox:
                x1, y1, x2, y2 = self.roi_bbox
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    input_image = frame_rgb[y1:y2, x1:x2].copy()
                    offset_x, offset_y = x1, y1
                else:
                    self.roi_active = False
                    
        # 포즈 추정 수행
        if self.roi_bbox or not self.tracker_type:
            try:
                input_image.flags.writeable = False
                results = self.pose.process(input_image)
                input_image.flags.writeable = True
            except Exception as e:
                print(f"[ERROR] MediaPipe 처리 실패: {e}")
                return None, output_frame
            
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    pixel_x_roi = landmark.x * input_image.shape[1]
                    pixel_y_roi = landmark.y * input_image.shape[0]
                    landmark.x = (pixel_x_roi + offset_x) / width
                    landmark.y = (pixel_y_roi + offset_y) / height
                    landmark.z = landmark.z * input_image.shape[1]
                    
                # if self.tracker_type == "yolotrack" and self.tracker:
                    # try:
                        # self.roi_bbox = self.tracker.adjust_roi(frame, results.pose_landmarks.landmark, self.former_roi)
                    # except Exception as e:
                        # print(f"[WARN] ROI 조정 실패: {e}")
                        
                if not self.tracker_type and self.tracker:
                    try:
                        self.roi_bbox = self.tracker._calculate_roi(frame, results.pose_landmarks.landmark)
                    except Exception:
                        self.roi_bbox = self.tracker._default_roi(frame)
                    
            else:
                if not self.tracker_type and self.tracker:
                    self.roi_bbox = self.tracker._default_roi(frame)
        else: return None, output_frame
                    
        # ROI 시각화
        if self.roi_bbox:
            x1, y1, x2, y2 = self.roi_bbox
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
        return results.pose_landmarks, output_frame

    def extract_3d_keypoints(self, landmarks, frame):
        if not landmarks:
            return []

        height, width, _ = frame.shape
        keypoints_3d, raw_z_values = [], []

        for i, lm in enumerate(landmarks.landmark):
            x, y, z = int(lm.x * width), int(lm.y * height), lm.z
            raw_z_values.append(z)
            keypoints_3d.append((x, y, z))

        if self.frame_count < self.initialization_frames:
            self.initial_z_values.append(raw_z_values)
            self.frame_count += 1
            if self.frame_count == self.initialization_frames:
                self.z_baseline = [np.median([f[i] for f in self.initial_z_values]) for i in range(len(raw_z_values))]
                print("Z 기준선 초기화 완료")

        return self._stabilize_z_values(keypoints_3d)

    def _stabilize_z_values(self, keypoints_3d):
        stabilized_keypoints = []

        for i, (x, y, z) in enumerate(keypoints_3d):
            if i not in self.z_history:
                self.z_history[i] = []

            if self.z_baseline is not None:
                z_diff = z - self.z_baseline[i]
                adjusted_z = self.z_baseline[i] + z_diff * self.z_scale_factor
                self.z_history[i].append(adjusted_z)
                if len(self.z_history[i]) > self.z_history_max_length:
                    self.z_history[i].pop(0)
                final_z = np.mean(self.z_history[i])
            else:
                final_z = z
                self.z_history[i].append(z)
                
                if len(self.z_history[i]) > self.z_history_max_length:
                    self.z_history[i].pop(0)
                
            stabilized_keypoints.append((x, y, final_z))

        return stabilized_keypoints
