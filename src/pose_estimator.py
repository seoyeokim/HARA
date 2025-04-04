# pose_estimator.py
import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 roi_ratio=0.8):  # 단일 ROI 비율
        """
        MediaPipe 포즈 추정기 초기화
        Args:
            min_detection_confidence (float): 최소 감지 신뢰도
            min_tracking_confidence (float): 최소 추적 신뢰도
            roi_ratio (float): ROI 크기 비율 (0.0 ~ 1.0) - 짧은 차원 기준
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

        self.roi_ratio = roi_ratio

    def _calculate_roi(self, frame):
        """
        프레임 중심에 짧은 차원을 기준으로 정사각형 ROI 계산
        Args:
            frame (numpy.ndarray): 입력 프레임
        Returns:
            tuple: ROI 좌표 (x1, y1, x2, y2)
        """
        height, width = frame.shape[:2]

        # 짧은 차원을 기준으로 ROI 크기 결정 (ROI 비율 적용)
        roi_size = int(min(width, height) * self.roi_ratio)

        # 화면 중심 계산
        center_x = width // 2
        center_y = height // 2

        # ROI 좌표 계산
        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(width, x1 + roi_size)
        y2 = min(height, y1 + roi_size)

        # 경계 검사 (ROI가 프레임을 벗어나는 경우 조정)
        if x2 > width:
            x1 = max(0, x1 - (x2 - width))
            x2 = width

        if y2 > height:
            y1 = max(0, y1 - (y2 - height))
            y2 = height

        return (x1, y1, x2, y2)

    def estimate_pose(self, frame):
        """
        프레임에서 포즈 추정 (ROI 내부)
        Args:
            frame (numpy.ndarray): 입력 프레임
        Returns:
            tuple: 포즈 랜드마크, ROI 표시된 프레임
        """
        # ROI 계산
        x1, y1, x2, y2 = self._calculate_roi(frame)

        # ROI 영역 추출
        roi_frame = frame[y1:y2, x1:x2].copy()

        # ROI 프레임 포즈 추정
        roi_frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(roi_frame_rgb)

        # 프레임에 ROI 시각화
        output_frame = frame.copy()
        # ROI 박스 그리기 코드 제거: cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 랜드마크 좌표 보정
        if results.pose_landmarks:
            roi_landmarks = results.pose_landmarks
            for landmark in roi_landmarks.landmark:
                # ROI 내 상대 좌표를 전체 프레임 좌표로 변환
                landmark.x = (landmark.x * (x2 - x1) + x1) / frame.shape[1]
                landmark.y = (landmark.y * (y2 - y1) + y1) / frame.shape[0]

        return results.pose_landmarks, output_frame

class PoseEstimator3D:
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 roi_ratio=0.8):  # 단일 ROI 비율
        """
        MediaPipe 3D 포즈 추정기 초기화
        Args:
            min_detection_confidence (float): 최소 감지 신뢰도
            min_tracking_confidence (float): 최소 추적 신뢰도
            roi_ratio (float): ROI 크기 비율 (0.0 ~ 1.0) - 짧은 차원 기준
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

        self.roi_ratio = roi_ratio

        # Z값 안정화를 위한 변수 추가
        self.z_history = {}  # 키포인트별 Z값 이력 저장
        self.z_history_max_length = 10  # 이력 길이 (프레임 수)
        self.z_baseline = None  # 기준 Z값 (초기화 후 설정)
        self.z_scale_factor = 0.5  # Z값 스케일 감소 (변화 줄이기)
        self.initialization_frames = 30  # 초기화에 사용할 프레임 수
        self.frame_count = 0  # 프레임 카운터
        self.initial_z_values = []  # 초기 Z값 저장용

    def _calculate_roi(self, frame):
        """
        프레임 중심에 짧은 차원을 기준으로 정사각형 ROI 계산
        Args:
            frame (numpy.ndarray): 입력 프레임
        Returns:
            tuple: ROI 좌표 (x1, y1, x2, y2)
        """
        height, width = frame.shape[:2]

        # 짧은 차원을 기준으로 ROI 크기 결정 (ROI 비율 적용)
        roi_size = int(min(width, height) * self.roi_ratio)

        # 화면 중심 계산
        center_x = width // 2
        center_y = height // 2

        # ROI 좌표 계산
        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(width, x1 + roi_size)
        y2 = min(height, y1 + roi_size)

        # 경계 검사 (ROI가 프레임을 벗어나는 경우 조정)
        if x2 > width:
            x1 = max(0, x1 - (x2 - width))
            x2 = width

        if y2 > height:
            y1 = max(0, y1 - (y2 - height))
            y2 = height

        return (x1, y1, x2, y2)

    def estimate_pose(self, frame):
        """
        프레임에서 3D 포즈 추정 (ROI 내부)
        Args:
            frame (numpy.ndarray): 입력 프레임
        Returns:
            tuple: 포즈 랜드마크, ROI 표시된 프레임
        """
        # ROI 계산
        x1, y1, x2, y2 = self._calculate_roi(frame)

        # ROI 영역 추출
        roi_frame = frame[y1:y2, x1:x2].copy()

        # ROI 프레임 포즈 추정
        roi_frame_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

        # ROI 영역의 차원 정보를 명시적으로 지정하기 위한 방법
        # (MediaPipe는 직접적인 IMAGE_DIMENSIONS 설정을 제공하지 않으므로 정사각형 ROI로 처리)
        results = self.pose.process(roi_frame_rgb)

        # 프레임에 ROI 시각화
        output_frame = frame.copy()
        # ROI 박스 그리기 코드 제거: cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 랜드마크 좌표 보정
        if results.pose_landmarks:
            roi_landmarks = results.pose_landmarks
            for landmark in roi_landmarks.landmark:
                # ROI 내 상대 좌표를 전체 프레임 좌표로 변환
                landmark.x = (landmark.x * (x2 - x1) + x1) / frame.shape[1]
                landmark.y = (landmark.y * (y2 - y1) + y1) / frame.shape[0]

        return results.pose_landmarks, output_frame

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
