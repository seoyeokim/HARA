# kalman_filter.py
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class KalmanFilterTracker:
    def __init__(self, num_keypoints=33):
        """
        다수의 키포인트를 위한 칼만 필터 초기화
        Args:
            num_keypoints (int): 추적할 키포인트 수
        """
        self.kalman_filters = {i: self._create_kalman_filter() for i in range(num_keypoints)}

    def _create_kalman_filter(self):
        """개별 칼만 필터 생성"""
        kf = cv2.KalmanFilter(4, 2)

        # 상태 전이 행렬
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)

        # 측정 행렬
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], np.float32)

        # 노이즈 설정
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.zeros((4, 1), dtype=np.float32)

        return kf

    def track(self, keypoints):
        """
        키포인트 추적 및 필터링
        Args:
            keypoints (list): 원본 키포인트 좌표 리스트
        Returns:
            list: 필터링된 키포인트 좌표
        """
        filtered_keypoints = []
        for i, (x, y) in enumerate(keypoints):
            kf = self.kalman_filters[i]
            measurement = np.array([[np.float32(x)], [np.float32(y)]])

            prediction = kf.predict()
            estimated = kf.correct(measurement)

            filtered_keypoints.append((int(estimated[0]), int(estimated[1])))

        return filtered_keypoints

class KalmanFilterTracker3D:
    def __init__(self, num_keypoints=33):
        """
        3D 키포인트를 위한 칼만 필터 초기화
        Args:
            num_keypoints (int): 추적할 키포인트 수
        """
        self.kalman_filters = {i: self._create_kalman_filter() for i in range(num_keypoints)}

    def _create_kalman_filter(self):
        """3D 개별 칼만 필터 생성"""
        # 상태 벡터 크기: 6 (x, y, z, dx, dy, dz)
        # 측정 벡터 크기: 3 (x, y, z)
        kf = cv2.KalmanFilter(6, 3)

        # 상태 전이 행렬 (3D 움직임 모델)
        kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]], np.float32)

        # 측정 행렬 (위치만 측정)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]], np.float32)

        # 노이즈 설정 (3D에 맞게 조정)
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.001
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.01
        kf.errorCovPost = np.eye(6, dtype=np.float32)
        kf.statePost = np.zeros((6, 1), dtype=np.float32)

        return kf

    def track(self, keypoints):
        """
        3D 키포인트 추적 및 필터링
        Args:
            keypoints (list): 원본 3D 키포인트 좌표 리스트
        Returns:
            list: 필터링된 3D 키포인트 좌표
        """
        filtered_keypoints = []
        for i, (x, y, z) in enumerate(keypoints):
            kf = self.kalman_filters[i]
            measurement = np.array([
                [np.float32(x)],
                [np.float32(y)],
                [np.float32(z)]
            ])

            prediction = kf.predict()
            estimated = kf.correct(measurement)

            filtered_keypoints.append((
                int(estimated[0]),
                int(estimated[1]),
                int(estimated[2])
            ))

        return filtered_keypoints

class KeypointPreprocess:
    def __init__(self, norm_type = None):
        self.norm_type = norm_type
        
    def _points_normalization(self, keypoint_matrix):
        norm_matrix = np.array([
            self.scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in keypoint_matrix
        ])
        
        return norm_matrix
        
    def _coordinate_convert(self, keypoints):
        converted_keypoints = []
        keypoints = np.array(keypoints)
        
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

        converted_keypoints = [(tuple((kp @ rotation_matrix) - middle_point)) for kp in keypoints]

        return converted_keypoints

    def process(self, keypoints):
        converted_keypoints = self._coordinate_convert(keypoints)

        if self.norm_type != None:
            preprosessed_keypoints = [tuple(kp) for kp in self._points_normalization(np.array(converted_keypoints))]
        else:
            preprosessed_keypoints = converted_keypoints
            
        return preprosessed_keypoints