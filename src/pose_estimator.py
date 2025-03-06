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
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

        for landmark in landmarks.landmark:
            # 2D 좌표 변환
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # 깊이 추정 (MediaPipe의 z 값 사용)
            # landmark.z는 MediaPipe에서 제공하는 상대적 깊이 값
            z = int(landmark.z * width)  # 깊이를 너비 기준으로 스케일링

            keypoints_3d.append((x, y, z))

        return keypoints_3d
