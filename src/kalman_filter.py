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
        """3D 개별 칼만 필터 생성 - Z값 안정화를 위한 파라미터 조정"""
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

        # Z축 안정화를 위한 노이즈 설정 조정
        process_noise = np.eye(6, dtype=np.float32) * 0.001
        # Z축 프로세스 노이즈 감소 (Z축 변화를 더 부드럽게)
        process_noise[2, 2] *= 0.1  # Z 위치에 대한 프로세스 노이즈 감소
        process_noise[5, 5] *= 0.1  # Z 속도에 대한 프로세스 노이즈 감소
        kf.processNoiseCov = process_noise

        measurement_noise = np.eye(3, dtype=np.float32) * 0.01
        # Z축 측정 노이즈 증가 (측정값보다 예측값을 더 신뢰)
        measurement_noise[2, 2] = 0.1  # Z 측정에 대한 노이즈 증가
        kf.measurementNoiseCov = measurement_noise

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

            # 예측 단계
            prediction = kf.predict()

            # 보정 단계
            estimated = kf.correct(measurement)

            # Z값에 추가 필터링 적용
            # 현재는 칼만 필터의 Z 파라미터 조정으로 충분하지만,
            # 필요 시 여기에 추가적인 Z값 필터링 로직 구현 가능

            filtered_keypoints.append((
                int(estimated[0]),
                int(estimated[1]),
                int(estimated[2])
            ))

        return filtered_keypoints

class KeypointPreprocess:
    def __init__(self, norm_type = None):
        self.scaler = MinMaxScaler()
        self.norm_type = norm_type

    def _points_normalization(self, keypoint_matrix):
        x_in_row, y_in_row, z_in_row, norm_matrix = [], [], [], []

        for row in keypoint_matrix:
            x_in_row.append(row[0])
            y_in_row.append(row[1])
            z_in_row.append(row[2])

        norm_x = self.scaler.fit_transform(np.array(x_in_row).reshape(-1, 1)).flatten() * 0.25
        norm_y = self.scaler.fit_transform(np.array(y_in_row).reshape(-1, 1)).flatten()
        norm_z = self.scaler.fit_transform(np.array(z_in_row).reshape(-1, 1)).flatten() * 0.15

        for x, y, z in zip(norm_x, norm_y, norm_z):
            norm_matrix.append([x, y, z])

        return np.array(norm_matrix)

    def _coordinate_convert(self, keypoints):
        converted_keypoints = []
        keypoints = np.array(keypoints)

        if len(keypoints) > 13:
            [x1, x2] = [keypoints[13], keypoints[12]]
            [y1, y2] = [keypoints[1], keypoints[0]]
        else:
            [x1, x2] = [keypoints[3], keypoints[2]]
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
