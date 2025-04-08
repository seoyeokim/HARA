# kalman_filter.py
import cv2
import numpy as np
import traceback
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

import numpy as np

class TFTKeypointPreprocess:
    """
    TFT 모델 입력을 위한 키포인트 전처리 클래스
    - 정규화 및 상대 좌표 변환 수행
    - 이상치 감지 및 보정 기능 추가
    """
    def __init__(self, normalization_type='z_score', reference_point='hip',
                 scale_normalize=True, include_temporal=False, window_size=5,
                 per_axis_normalization=True, outlier_correction=True,
                 outlier_threshold=5.0, correction_method='interpolate'):
        """
        키포인트 전처리 초기화

        Args:
            normalization_type (str): 정규화 유형 ('z_score', 'min_max', 'none')
            reference_point (str): 상대 좌표 변환 기준점 ('com', 'hip')
            scale_normalize (bool): 척도 정규화 여부
            include_temporal (bool): 시간적 특성(속도, 가속도) 포함 여부
            window_size (int): 시간적 특성 계산을 위한 윈도우 크기
            per_axis_normalization (bool): 축(x, y, z)별 독립 정규화 여부
            outlier_correction (bool): 이상치 보정 여부
            outlier_threshold (float): 이상치 탐지를 위한 표준편차 임계값
            correction_method (str): 이상치 보정 방법 ('clip', 'median', 'interpolate')
        """
        self.normalization_type = normalization_type
        self.reference_point = reference_point
        self.scale_normalize = scale_normalize
        self.include_temporal = include_temporal
        self.window_size = window_size
        self.per_axis_normalization = per_axis_normalization

        # 이상치 보정 관련 설정
        self.outlier_correction = outlier_correction
        self.outlier_threshold = outlier_threshold
        self.correction_method = correction_method

        # 이동 평균 필터를 위한 버퍼
        self.keypoints_buffer = []

        # 이상치 보정을 위한 통계 추적
        self.outlier_stats = {
            'detected': 0,  # 감지된 이상치 수
            'corrected': 0  # 보정된 이상치 수
        }

        # 정규화를 위한 통계 정보
        if per_axis_normalization:
            # 축별 독립 정규화를 위한 통계
            self.means = np.zeros(3)  # [x_mean, y_mean, z_mean]
            self.stds = np.ones(3)    # [x_std, y_std, z_std]
            self.mins = np.zeros(3)   # [x_min, y_min, z_min]
            self.maxs = np.ones(3)    # [x_max, y_max, z_max]
            self.stats_initialized = False
        else:
            # 기존 통합 정규화를 위한 통계
            self.means = None
            self.stds = None
            self.mins = None
            self.maxs = None
            self.stats_initialized = False

    def process(self, keypoints_and_com):
        """
        키포인트 및 CoM 전처리 수행

        Args:
            keypoints_and_com (list): 키포인트 및 CoM 좌표 리스트
                                      [keypoint_11, keypoint_12, ..., com]
                                      각 요소는 [x, y, z] 형태의 리스트

        Returns:
            list: 전처리된 키포인트 및 CoM 좌표 리스트
        """
        # NumPy 배열로 변환
        keypoints_array = np.array(keypoints_and_com[:-1])  # CoM 제외한 키포인트
        com = np.array(keypoints_and_com[-1])               # CoM

        # 1. 상대 좌표 변환
        keypoints_relative = self._convert_to_relative_coords(keypoints_array, com)

        # 2. 척도 정규화 (선택적)
        if self.scale_normalize:
            keypoints_normalized = self._apply_scale_normalization(keypoints_relative)
        else:
            keypoints_normalized = keypoints_relative

        # 3. 이상치 보정 (선택적)
        if self.outlier_correction:
            keypoints_normalized = self._correct_outliers(keypoints_normalized)

        # 4. 통계적 정규화 (Z-점수 또는 Min-Max)
        processed_keypoints = self._apply_normalization(keypoints_normalized)

        # 5. 시간적 특성 추출 (선택적)
        if self.include_temporal and len(self.keypoints_buffer) >= self.window_size:
            processed_keypoints = self._extract_temporal_features(processed_keypoints)

        # 6. 이동 평균 필터링을 위한 버퍼 업데이트
        self._update_buffer(keypoints_normalized)

        # NumPy 배열을 리스트로 변환하여 반환
        result = processed_keypoints.tolist()

        # CoM 좌표 처리
        if self.reference_point == 'com':
            # CoM 기준이면 CoM은 항상 [0,0,0]
            com_processed = np.zeros(3).tolist()
        else:
            # Hip 기준인 경우 CoM의 상대 좌표 계산
            # MediaPipe에서 hip은 23, 24번(인덱스 12, 13)
            left_hip_index = 12  # 23번 키포인트 (11번 이후 기준)
            right_hip_index = 13  # 24번 키포인트 (11번 이후 기준)

            try:
                hip_center = (keypoints_array[left_hip_index] + keypoints_array[right_hip_index]) / 2
                com_relative = com - hip_center

                # CoM도 키포인트와 동일하게 정규화 적용
                if self.scale_normalize:
                    # 키포인트와 동일한 척도 정규화 적용
                    reference_distance = self._get_reference_distance(keypoints_array)
                    com_normalized = com_relative / (reference_distance if reference_distance > 1e-6 else 1.0)
                else:
                    com_normalized = com_relative

                # 이상치 보정 적용 (CoM도 보정 대상에 포함)
                if self.outlier_correction:
                    com_normalized = self._correct_outliers(com_normalized.reshape(1, 3)).flatten()

                # 단일 좌표를 (1, 3) 형태로 변환하여 정규화
                com_processed = self._apply_normalization(com_normalized.reshape(1, 3)).flatten().tolist()
            except (IndexError, ValueError) as e:
                print(f"Warning: Hip 키포인트 인덱스 오류 - {e}. CoM 좌표를 [0,0,0]으로 설정합니다.")
                com_processed = np.zeros(3).tolist()

        result.append(com_processed)
        return result

    def _correct_outliers(self, keypoints):
        """
        이상치 감지 및 보정

        Args:
            keypoints (numpy.ndarray): 키포인트 좌표 배열

        Returns:
            numpy.ndarray: 이상치가 보정된 키포인트 배열
        """
        original_shape = keypoints.shape
        corrected_keypoints = keypoints.copy()

        # 축별로 이상치 보정 수행
        for axis in range(3):  # x, y, z 축
            axis_values = keypoints[..., axis].flatten()

            # 축별 통계 계산
            median = np.median(axis_values)
            q1, q3 = np.percentile(axis_values, [25, 75])
            iqr = q3 - q1

            # IQR 방식으로 이상치 범위 정의
            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr

            # Z-점수 방식으로 이상치 범위 정의 (대안)
            mean = np.mean(axis_values)
            std = np.std(axis_values)
            z_lower_bound = mean - self.outlier_threshold * std
            z_upper_bound = mean + self.outlier_threshold * std

            # 최종 이상치 범위는 두 방식 중 더 넓은 범위를 선택
            final_lower = min(lower_bound, z_lower_bound)
            final_upper = max(upper_bound, z_upper_bound)

            # 이상치 마스크 생성
            outlier_mask = (keypoints[..., axis] < final_lower) | (keypoints[..., axis] > final_upper)
            outlier_count = np.sum(outlier_mask)
            self.outlier_stats['detected'] += outlier_count

            if outlier_count > 0:
                if self.correction_method == 'clip':
                    # 방법 1: 임계값으로 클리핑
                    corrected_keypoints[..., axis] = np.clip(
                        corrected_keypoints[..., axis],
                        final_lower,
                        final_upper
                    )

                elif self.correction_method == 'median':
                    # 방법 2: 중앙값으로 대체
                    corrected_keypoints[outlier_mask, axis] = median

                elif self.correction_method == 'interpolate' and len(self.keypoints_buffer) > 0:
                    # 방법 3: 이전 프레임 값으로 보간 (시간적 보간)
                    # 참고: 이 방법은 키포인트 버퍼에 이전 프레임이 있을 때만 사용 가능
                    for idx in np.where(outlier_mask.flatten())[0]:
                        # 인덱스를 다차원 인덱스로 변환
                        multi_idx = np.unravel_index(idx, keypoints[..., axis].shape)

                        # 이전 프레임들의 해당 키포인트 값 추출
                        prev_values = [
                            buf[multi_idx][axis] for buf in self.keypoints_buffer[-3:]
                            if len(buf) > multi_idx[0]
                        ]

                        if prev_values:
                            # 이전 값들의 평균으로 대체
                            corrected_keypoints[multi_idx][axis] = np.mean(prev_values)
                        else:
                            # 이전 값이 없으면 중앙값으로 대체
                            corrected_keypoints[multi_idx][axis] = median

                self.outlier_stats['corrected'] += outlier_count
                print(f"축 {axis}에서 {outlier_count}개의 이상치 감지 및 보정 (범위: {final_lower:.4f} ~ {final_upper:.4f})")

        return corrected_keypoints

    def _convert_to_relative_coords(self, keypoints, reference):
        """
        기준점(CoM 또는 Hip) 대비 상대 좌표로 변환

        Args:
            keypoints (numpy.ndarray): 키포인트 좌표 배열
            reference (numpy.ndarray): 기준점 좌표 (CoM)

        Returns:
            numpy.ndarray: 상대 좌표로 변환된 키포인트 배열
        """
        if self.reference_point == 'com':
            # CoM 기준 상대 좌표
            return keypoints - reference
        else:
            # Hip 중심점 기준 상대 좌표
            # MediaPipe Pose에서 hip은 23, 24번 키포인트 (11번부터 시작할 경우 인덱스 12, 13)
            left_hip_index = 12  # 23번 키포인트 (11번 이후 기준)
            right_hip_index = 13  # 24번 키포인트 (11번 이후 기준)

            try:
                hip_center = (keypoints[left_hip_index] + keypoints[right_hip_index]) / 2
                return keypoints - hip_center
            except IndexError:
                print("Warning: Hip 키포인트 인덱스(12, 13)를 찾을 수 없습니다. 첫 두 개의 키포인트를 사용합니다.")
                # 인덱스 오류 발생 시 대체 로직
                hip_center = (keypoints[0] + keypoints[1]) / 2
                return keypoints - hip_center

    def _get_reference_distance(self, keypoints):
        """
        정규화를 위한 참조 거리 계산 (hip 너비)

        Args:
            keypoints (numpy.ndarray): 키포인트 좌표 배열

        Returns:
            float: 참조 거리 (hip 너비)
        """
        try:
            # MediaPipe에서 hip은 23, 24번 키포인트 (11번부터 시작할 경우 인덱스 12, 13)
            left_hip_index = 12
            right_hip_index = 13

            # hip 너비 계산
            hip_width = np.linalg.norm(keypoints[left_hip_index] - keypoints[right_hip_index])

            # hip 너비가 너무 작은 경우(이상치) 대체값 사용
            if hip_width < 1e-2:
                print(f"Warning: Hip 너비({hip_width})가 너무 작습니다. 기본값 1.0을 사용합니다.")
                return 1.0

            return hip_width if hip_width > 1e-6 else 1.0
        except IndexError:
            print("Warning: Hip 키포인트 인덱스를 찾을 수 없습니다. 기본값 1.0을 사용합니다.")
            return 1.0

    def _apply_scale_normalization(self, keypoints):
        """
        인체 크기에 대한 정규화 적용

        Args:
            keypoints (numpy.ndarray): 키포인트 좌표 배열

        Returns:
            numpy.ndarray: 크기 정규화된 키포인트 배열
        """
        reference_distance = self._get_reference_distance(keypoints)
        return keypoints / reference_distance

    def _apply_normalization(self, keypoints):
        """
        통계적 정규화 적용 (Z-점수 또는 Min-Max)

        축별 독립 정규화 또는 통합 정규화 선택적 적용

        Args:
            keypoints (numpy.ndarray): 키포인트 좌표 배열

        Returns:
            numpy.ndarray: 정규화된 키포인트 배열
        """
        if self.normalization_type == 'none':
            return keypoints

        original_shape = keypoints.shape

        if self.per_axis_normalization:
            # 축별 독립 정규화 적용
            normalized = np.zeros_like(keypoints)

            if not self.stats_initialized:
                # 처음 호출 시 축별 통계 계산
                for i in range(3):  # x, y, z 축
                    axis_values = keypoints[..., i]

                    if self.normalization_type == 'z_score':
                        self.means[i] = np.mean(axis_values)
                        self.stds[i] = np.std(axis_values)
                        # 표준편차가 0이면 1로 설정 (0으로 나누기 방지)
                        if self.stds[i] < 1e-6:
                            self.stds[i] = 1.0

                    elif self.normalization_type == 'min_max':
                        self.mins[i] = np.min(axis_values)
                        self.maxs[i] = np.max(axis_values)
                        # 최소값과 최대값이 같으면 나누기 방지
                        if np.abs(self.maxs[i] - self.mins[i]) < 1e-6:
                            self.maxs[i] = self.mins[i] + 1.0

                self.stats_initialized = True

            # 축별로 정규화 적용
            for i in range(3):  # x, y, z 축
                if self.normalization_type == 'z_score':
                    normalized[..., i] = (keypoints[..., i] - self.means[i]) / self.stds[i]
                elif self.normalization_type == 'min_max':
                    normalized[..., i] = (keypoints[..., i] - self.mins[i]) / (self.maxs[i] - self.mins[i])

            return normalized

        else:
            # 기존 통합 정규화 적용
            flattened = keypoints.reshape(-1)

            if not self.stats_initialized:
                if self.normalization_type == 'z_score':
                    self.means = np.mean(flattened)
                    self.stds = np.std(flattened)
                    # 표준편차가 0이면 1로 설정 (0으로 나누기 방지)
                    if self.stds < 1e-6:
                        self.stds = 1.0

                elif self.normalization_type == 'min_max':
                    self.mins = np.min(flattened)
                    self.maxs = np.max(flattened)
                    # 최소값과 최대값이 같으면 나누기 방지
                    if np.abs(self.maxs - self.mins) < 1e-6:
                        self.maxs = self.mins + 1.0

                self.stats_initialized = True

            if self.normalization_type == 'z_score':
                normalized = (flattened - self.means) / self.stds
            elif self.normalization_type == 'min_max':
                normalized = (flattened - self.mins) / (self.maxs - self.mins)
            else:
                # 알 수 없는 정규화 유형
                print(f"Warning: 알 수 없는 정규화 유형 '{self.normalization_type}'. 원본 데이터를 반환합니다.")
                return keypoints

            # 원래 형태로 복원
            return normalized.reshape(original_shape)

    def _extract_temporal_features(self, current_keypoints):
        """
        시간적 특성(속도, 가속도) 추출

        Args:
            current_keypoints (numpy.ndarray): 현재 프레임의 키포인트

        Returns:
            numpy.ndarray: 시간적 특성이 추가된 키포인트
        """
        # 버퍼의 마지막 n개 프레임 사용
        recent_frames = self.keypoints_buffer[-self.window_size:]
        recent_frames.append(current_keypoints)
        frames_array = np.array(recent_frames)

        # 속도: 연속된 프레임 간 차이
        velocities = np.diff(frames_array, axis=0)

        # 가속도: 속도의 차이
        accelerations = np.diff(velocities, axis=0)

        # 현재 키포인트, 속도, 가속도 결합
        # 각 키포인트에 대해 [x, y, z, vx, vy, vz, ax, ay, az] 형태로 확장
        current_velocity = velocities[-1]
        current_acceleration = accelerations[-1]

        # 각 키포인트마다 위치, 속도, 가속도를 결합
        result = []
        for i in range(len(current_keypoints)):
            keypoint_features = np.concatenate([
                current_keypoints[i],
                current_velocity[i],
                current_acceleration[i]
            ])
            result.append(keypoint_features)

        return np.array(result)

    def _update_buffer(self, keypoints):
        """
        이동 평균 필터링을 위한 버퍼 업데이트

        Args:
            keypoints (numpy.ndarray): 현재 프레임의 키포인트
        """
        self.keypoints_buffer.append(keypoints)

        # 버퍼 크기 제한
        if len(self.keypoints_buffer) > self.window_size + 2:
            self.keypoints_buffer.pop(0)

    def reset(self):
        """
        전처리기 상태 리셋
        """
        self.keypoints_buffer = []
        if self.per_axis_normalization:
            self.means = np.zeros(3)
            self.stds = np.ones(3)
            self.mins = np.zeros(3)
            self.maxs = np.ones(3)
        else:
            self.means = None
            self.stds = None
            self.mins = None
            self.maxs = None
        self.stats_initialized = False

        # 이상치 통계 리셋
        self.outlier_stats = {
            'detected': 0,
            'corrected': 0
        }

    def get_outlier_stats(self):
        """
        이상치 처리 통계 반환

        Returns:
            dict: 이상치 감지 및 보정 통계
        """
        return self.outlier_stats
