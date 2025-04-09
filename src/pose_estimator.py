# pose_estimator.py
import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator3D:
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 roi_padding=80,
                 roi_ratio=0.8):
        """
        MediaPipe 3D 포즈 추정기 초기화 (동적 ROI 지원)
        Args:
            min_detection_confidence (float): 최소 감지 신뢰도
            min_tracking_confidence (float): 최소 추적 신뢰도
            roi_padding (int): 감지된 랜드마크 주변 ROI 여유 공간 (픽셀)
        """
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.default_roi_bbox = None # 기본 ROI 좌표 저장용
        self.roi_bbox = None         # 현재(동적) ROI 좌표 저장용
        self.roi_active = False      # 동적 ROI 활성 상태
        self.initial_detection_done = False # 첫 탐지 성공 여부 플래그
        self.roi_padding = roi_padding # 픽셀 단위 패딩
        self.roi_ratio = roi_ratio # 기본 ROI 크기 비율

        # Z값 안정화를 위한 변수 추가
        self.z_history = {}  # 키포인트별 Z값 이력 저장
        self.z_history_max_length = 10  # 이력 길이 (프레임 수)
        self.z_baseline = None  # 기준 Z값 (초기화 후 설정)
        self.z_scale_factor = 0.5  # Z값 스케일 감소 (변화 줄이기)
        self.initialization_frames = 30  # 초기화에 사용할 프레임 수
        self.frame_count = 0  # 프레임 카운터
        self.initial_z_values = []  # 초기 Z값 저장용

    def _calculate_default_roi(self, width, height):
        """프레임 크기 기반으로 기본 ROI(화면 중앙) 계산"""

        # 짧은 차원을 기준으로 ROI 크기 결정
        roi_size = int(min(width, height) * self.roi_ratio)

        # 화면 중심 계산
        center_x = width // 2
        center_y = height // 2

        # ROI 좌표 계산 (정수 변환)
        x1 = int(max(0, center_x - roi_size // 2))
        y1 = int(max(0, center_y - roi_size // 2))
        x2 = int(min(width, x1 + roi_size)) # x1 기준 크기 더하기
        y2 = int(min(height, y1 + roi_size)) # y1 기준 크기 더하기

        return (x1, y1, x2, y2)

    def _calculate_roi(self, landmarks, frame):
        """
        랜드마크 기반으로 ROI 계산
        Args:
            landmarks (list): MediaPipe Landmark 객체 리스트
            frame (numpy.ndarray): 입력 프레임
        Returns:
            tuple: 계산된 ROI 좌표 (x_min, y_min, x_max, y_max) 픽셀
        """
        height, width = frame.shape[:2]

        
        # 랜드마크가 없거나 빈 리스트인 경우 기본 ROI 계산
        if not landmarks:
            # print("No landmarks provided, calculating default ROI.") # 디버깅용
            return self._calculate_default_roi(width, height)

        # Normalized 좌표를 픽셀 좌표로 변환
        x_values = [lm.x * width for lm in landmarks]
        y_values = [lm.y * height for lm in landmarks]
            
        # 랜드마크 기반 ROI 좌표 계산
        x_min = min(x_values)
        y_min = min(y_values)
        x_max = max(x_values)
        y_max = max(y_values)

        # 패딩 추가
        x_min = int(x_min - self.roi_padding)
        y_min = int(y_min - self.roi_padding)
        x_max = int(x_max + self.roi_padding)
        y_max = int(y_max + self.roi_padding)

        # 경계 검사 (ROI가 프레임을 벗어나는 경우 조정)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        # 유효한 박스인지 확인 (넓이/높이가 0보다 커야 함)
        if x_min < x_max and y_min < y_max:
            return (x_min, y_min, x_max, y_max)
        else:
            return self._calculate_default_roi(width, height)

    def estimate_pose(self, frame):
        """
        동적 ROI 기반 포즈 추정
        Args:
            frame (numpy.ndarray): 입력 프레임
        Returns:
            tuple: 포즈 랜드마크, ROI 표시된 프레임
        """
        height, width = frame.shape[:2]
        output_frame = frame.copy() # 결과용 프레임 복사
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.default_roi_bbox is None:
            self.default_roi_bbox = self._calculate_default_roi(width, height)
            self.roi_bbox = self.default_roi_bbox

        current_process_bbox = None
        
        if not self.initial_detection_done:
            current_process_bbox = self.default_roi_bbox
            process_in_roi = True # 기본 ROI 내에서 처리
            # print("Using DEFAULT ROI (Initial)") # 디버깅용
        elif self.roi_active and self.roi_bbox:
            # 첫 탐지 성공했고, 동적 ROI가 활성화 상태면 동적 ROI 사용
            current_process_bbox = self.roi_bbox
            process_in_roi = True # 동적 ROI 내에서 처리
            # print("Using DYNAMIC ROI") # 디버깅용
        else:
            # 첫 탐지는 성공했으나, 이전 프레임에서 실패하여 roi_active가 False인 경우
            current_process_bbox = self.default_roi_bbox
            self.roi_bbox = self.default_roi_bbox
            process_in_roi = True # 기본 ROI 내에서 처리
        
        # ROI용 이미지 초기 설정
        input_image = frame_rgb
        offset_x, offset_y = 0, 0
        
        if process_in_roi and current_process_bbox:
            x1, y1, x2, y2 = map(int, current_process_bbox)
            # 프레임 범위 내로 ROI 좌표 조정
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

        if x1 < x2 and y1 < y2:
            input_image = frame_rgb[y1:y2, x1:x2].copy()
            offset_x, offset_y = x1, y1
        else:
            # 계산된 ROI가 유효하지 않은 경우
            input_image = frame_rgb
            process_in_roi = False
            self.roi_active = False

        # MediaPipe 포즈 처리
        results = None
        try:
            # 성능 향상을 위해 이미지 쓰기 불가 설정
            input_image.flags.writeable = False
            results = self.pose.process(input_image)
            input_image.flags.writeable = True
        except Exception as e:
            print(f"Error during MediaPipe pose processing: {e}")
            # 에러 발생 시 ROI 비활성화하여 다음 프레임은 전체 처리 시도
            self.roi_active = False
            self.roi_bbox = self._calculate_default_roi(width, height)


        # 결과 처리 및 다음 ROI 계산
        corrected_landmarks_object = None # 반환할 랜드마크 보정 객체

        if results and results.pose_landmarks:
            roi_landmarks = results.pose_landmarks # ROI 기준 랜드마크
            roi_h, roi_w = input_image.shape[:2] # 처리된 이미지(ROI 또는 전체)의 크기

            # 랜드마크 좌표 보정
            for landmark in roi_landmarks.landmark:
                if not self.initial_detection_done:
                    self.initial_detection_done = True
                # 1. ROI 내 픽셀 좌표 계산
                pixel_x_roi = landmark.x * roi_w
                pixel_y_roi = landmark.y * roi_h
                # 2. 전체 프레임 내 픽셀 좌표 계산
                pixel_x_frame = pixel_x_roi + offset_x
                pixel_y_frame = pixel_y_roi + offset_y
                # 3. 전체 프레임 기준 Normalized 좌표로 변환
                landmark.x = pixel_x_frame / width
                landmark.y = pixel_y_frame / height
                # landmark.z = landmark.z * roi_w  # 너비 기준 z좌표 보정 (선택사항)

            corrected_landmarks_object = roi_landmarks # 보정된 랜드마크 객체 저장

            # 다음 프레임을 위한 새로운 ROI 계산
            # 보정된 랜드마크 사용
            new_roi = self._calculate_roi(corrected_landmarks_object.landmark, frame)

            if new_roi:
                self.roi_bbox = new_roi
                self.roi_active = True
            else:
                # 랜드마크는 감지했지만 ROI 계산 실패한 경우
                self.roi_active = False
                self.roi_bbox = self._calculate_default_roi(width, height)

        else:
            # 랜드마크 감지 실패
            if process_in_roi:
                self.roi_active = False
                self.roi_bbox = self._calculate_default_roi(width, height)

        # 시각화
        # ROI 영역 시각화 (감지된 경우, 파란색 박스)
        if self.roi_active and self.roi_bbox:
             x1, y1, x2, y2 = self.roi_bbox
             cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        elif not self.roi_active and self.roi_bbox:  # ROI 비활성 상태 표시 (self.roi_bbox 확인 추가)
             x1, y1, x2, y2 = self.roi_bbox
             cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
             cv2.putText(output_frame, "ROI Inactive", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             
        return corrected_landmarks_object, output_frame
    
    def extract_3d_keypoints(self, landmarks, frame):
        """
        3D 키포인트 추출 (깊이 추정 포함)
        Args:
            landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): 랜드마크
            frame (numpy.ndarray): 입력 프레임
        Returns:
            list: 3D 키포인트 좌표
        """
        if not landmarks:
            return []

        height, width, _ = frame.shape

        # 3D 키포인트 리스트 (x, y, z)
        keypoints_3d = []
        raw_z_values = []

        for i, landmark in enumerate(landmarks.landmark):
            # 2D 좌표 변환
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # 원시 Z값 획득 (아직 처리하지 않음)
            raw_z = landmark.z * width  # 깊이를 너비 기준으로 스케일링
            raw_z_values.append(raw_z)

            # 임시로 원시 Z값 사용 (이후에 안정화된 값으로 교체)
            keypoints_3d.append((x, y, raw_z))

        # 초기화 단계: 처음 N프레임 동안 기준 Z값 계산
        if self.frame_count < self.initialization_frames:
            self.initial_z_values.append(raw_z_values)
            self.frame_count += 1

            # 초기화 완료 시 기준 Z값 설정
            if self.frame_count == self.initialization_frames:
                # 각 키포인트별로 초기 프레임의 중간값 계산
                z_medians = []
                for i in range(len(raw_z_values)):
                    keypoint_z_values = [frame_data[i] for frame_data in self.initial_z_values]
                    z_medians.append(np.median(keypoint_z_values))

                self.z_baseline = z_medians
                print("Z값 기준선 초기화 완료")

        # Z값 안정화 처리
        stabilized_keypoints_3d = self._stabilize_z_values(keypoints_3d)

        return stabilized_keypoints_3d

    def _stabilize_z_values(self, keypoints_3d):
        """
        Z값 안정화 처리 메서드
        Args:
            keypoints_3d: 원시 3D 키포인트 리스트
        Returns:
            list: 안정화된 3D 키포인트 리스트
        """
        stabilized_keypoints = []

        for i, (x, y, z) in enumerate(keypoints_3d):
            # 키포인트 히스토리 초기화 (없는 경우)
            if i not in self.z_history:
                self.z_history[i] = []

            # 기준선이 설정되었으면 Z값 보정
            if self.z_baseline is not None:
                # 기준 Z값과의 차이 계산 (변화량)
                z_diff = z - self.z_baseline[i]

                # 변화량 스케일 조정 (감소)
                scaled_z_diff = z_diff * self.z_scale_factor

                # 새로운 Z값 = 기준값 + 조정된 변화량
                adjusted_z = self.z_baseline[i] + scaled_z_diff

                # 이력에 추가
                self.z_history[i].append(adjusted_z)

                # 이력 길이 제한
                if len(self.z_history[i]) > self.z_history_max_length:
                    self.z_history[i].pop(0)

                # 이동 평균 계산 (최근 N개 프레임)
                final_z = np.mean(self.z_history[i])
            else:
                # 초기화 전에는 원시 Z값 그대로 사용
                final_z = z
                self.z_history[i].append(z)

                # 이력 길이 제한
                if len(self.z_history[i]) > self.z_history_max_length:
                    self.z_history[i].pop(0)

            stabilized_keypoints.append((x, y, final_z))

        return stabilized_keypoints