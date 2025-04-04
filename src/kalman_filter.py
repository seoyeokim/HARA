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

'''
class TFTKeypointPreprocess:
    def __init__(self, norm_type = None):
        self.scaler = MinMaxScaler()
        self.norm_type = norm_type
        # 디버깅 로그 활성화
        self.debug_log_enabled = True

    def _debug_log(self, message):
        """디버깅 로그 출력 함수"""
        if self.debug_log_enabled:
            print(message)

    def _points_normalization(self, keypoint_matrix):
        """
        키포인트 정규화 함수 - 오류 강화 처리 추가

        Args:
            keypoint_matrix: 정규화할 키포인트 행렬

        Returns:
            정규화된 키포인트 행렬
        """
        x_in_row, y_in_row, z_in_row, norm_matrix = [], [], [], []

        try:
            # 잘못된 데이터 유형 확인
            if not isinstance(keypoint_matrix, np.ndarray):
                self._debug_log(f"  경고: keypoint_matrix가 numpy 배열이 아닙니다. 변환 시도. 타입: {type(keypoint_matrix)}")
                keypoint_matrix = np.array(keypoint_matrix)

            # 빈 행렬 검사
            if keypoint_matrix.size == 0:
                self._debug_log("  오류: 빈 keypoint_matrix")
                return np.array([])

            # 각 행에서 x, y, z 좌표 추출
            for row in keypoint_matrix:
                if len(row) >= 3:
                    x_in_row.append(row[0])
                    y_in_row.append(row[1])
                    z_in_row.append(row[2])
                else:
                    self._debug_log(f"  경고: 행 길이가 3보다 작음: {len(row)}")
                    x_in_row.append(0.0)
                    y_in_row.append(0.0)
                    z_in_row.append(0.0)

            # 스케일링 전에 배열이 비어있는지 확인
            if not x_in_row or not y_in_row or not z_in_row:
                self._debug_log("  오류: 좌표 배열이 비어 있습니다")
                return np.zeros((len(keypoint_matrix), 3))

            # NaN 값 확인 및 제거
            x_array = np.array(x_in_row).reshape(-1, 1)
            y_array = np.array(y_in_row).reshape(-1, 1)
            z_array = np.array(z_in_row).reshape(-1, 1)

            # NaN 값 확인
            if np.isnan(x_array).any() or np.isnan(y_array).any() or np.isnan(z_array).any():
                self._debug_log("  경고: NaN 값 감지, 0으로 대체")
                x_array = np.nan_to_num(x_array)
                y_array = np.nan_to_num(y_array)
                z_array = np.nan_to_num(z_array)

            # 무한값 확인
            if np.isinf(x_array).any() or np.isinf(y_array).any() or np.isinf(z_array).any():
                self._debug_log("  경고: 무한값 감지, 0으로 대체")
                x_array = np.nan_to_num(x_array, nan=0.0, posinf=0.0, neginf=0.0)
                y_array = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
                z_array = np.nan_to_num(z_array, nan=0.0, posinf=0.0, neginf=0.0)

            # 스케일링 적용
            try:
                norm_x = self.scaler.fit_transform(x_array).flatten() * 0.25
                norm_y = self.scaler.fit_transform(y_array).flatten()
                norm_z = self.scaler.fit_transform(z_array).flatten() * 0.15
            except Exception as e:
                self._debug_log(f"  스케일링 오류: {e}")
                # 오류 발생 시 단순 정규화 시도 (0~1)
                x_min, x_max = np.min(x_array), np.max(x_array)
                y_min, y_max = np.min(y_array), np.max(y_array)
                z_min, z_max = np.min(z_array), np.max(z_array)

                # min-max 정규화 직접 구현 (0 나누기 방지)
                x_range = x_max - x_min
                y_range = y_max - y_min
                z_range = z_max - z_min

                norm_x = ((x_array - x_min) / (x_range if x_range != 0 else 1)).flatten() * 0.25
                norm_y = ((y_array - y_min) / (y_range if y_range != 0 else 1)).flatten()
                norm_z = ((z_array - z_min) / (z_range if z_range != 0 else 1)).flatten() * 0.15

            # 정규화된 좌표로 행렬 구성
            for x, y, z in zip(norm_x, norm_y, norm_z):
                norm_matrix.append([x, y, z])

            return np.array(norm_matrix)

        except Exception as e:
            self._debug_log(f"  정규화 중 예외 발생: {e}")
            # 오류 발생 시 원본 크기의 0 행렬 반환
            return np.zeros((len(keypoint_matrix) if hasattr(keypoint_matrix, '__len__') else 0, 3))

    def _coordinate_convert(self, keypoints):
        """
        좌표계 변환 함수 - 안전성 향상

        Args:
            keypoints: 변환할 키포인트 리스트

        Returns:
            변환된 키포인트 리스트
        """
        try:
            converted_keypoints = []

            # 키포인트 타입 확인 및 변환
            if not isinstance(keypoints, np.ndarray):
                self._debug_log(f"  키포인트가 numpy 배열이 아님, 변환 시도. 타입: {type(keypoints)}")
                keypoints = np.array(keypoints)

            # 빈 배열 검사
            if keypoints.size == 0:
                self._debug_log("  빈 keypoints 배열")
                return []

            # 키포인트 인덱스 유효성 검사
            if len(keypoints) > 13:
                # x 벡터에 대한 키포인트 (어깨 키포인트)
                x1_idx, x2_idx = 13, 12
                # y 벡터에 대한 키포인트 (눈 키포인트)
                y1_idx, y2_idx = 1, 0
            else:
                # 인덱스 부족 시 사용 가능한 인덱스로 조정
                self._debug_log(f"  키포인트 인덱스 부족, 조정 (개수: {len(keypoints)})")
                x1_idx, x2_idx = min(3, len(keypoints) - 1), min(2, len(keypoints) - 1)
                y1_idx, y2_idx = min(1, len(keypoints) - 1), min(0, len(keypoints) - 1)

            # 인덱스 범위 벗어남 방지
            if x1_idx >= len(keypoints) or x2_idx >= len(keypoints) or y1_idx >= len(keypoints) or y2_idx >= len(keypoints):
                self._debug_log("  인덱스 범위 초과, 기본 좌표계 사용")
                return [tuple(kp) for kp in keypoints]  # 변환 없이 반환

            # 좌표계 변환을 위한 벡터 설정
            x1, x2 = keypoints[x1_idx], keypoints[x2_idx]
            y1, y2 = keypoints[y1_idx], keypoints[y2_idx]

            # NaN 또는 무한값 확인
            if (np.isnan(x1).any() or np.isnan(x2).any() or np.isnan(y1).any() or np.isnan(y2).any() or
                np.isinf(x1).any() or np.isinf(x2).any() or np.isinf(y1).any() or np.isinf(y2).any()):
                self._debug_log("  좌표에 NaN 또는 무한값 감지, 기본 좌표계 사용")
                return [tuple(kp) for kp in keypoints]  # 변환 없이 반환

            s_vector = (y1 - x1) + (y2 - x2)

            # x 방향 벡터 (어깨선 방향)
            x_vector = (x2 - x1)
            x_norm = np.linalg.norm(x_vector)
            if x_norm < 1e-10:  # 0으로 나누기 방지
                self._debug_log("  x_vector 정규화 불가 (길이가 0에 가까움), 기본 좌표계 사용")
                return [tuple(kp) for kp in keypoints]  # 변환 없이 반환
            x_vector = x_vector / x_norm

            # z 방향 벡터 (x와 s의 외적)
            z_vector = np.cross(x_vector, s_vector)
            z_norm = np.linalg.norm(z_vector)
            if z_norm < 1e-10:  # 0으로 나누기 방지
                self._debug_log("  z_vector 정규화 불가 (길이가 0에 가까움), 기본 좌표계 사용")
                return [tuple(kp) for kp in keypoints]  # 변환 없이 반환
            z_vector /= z_norm

            # y 방향 벡터 (z와 x의 외적)
            y_vector = np.cross(z_vector, x_vector)
            y_norm = np.linalg.norm(y_vector)
            if y_norm < 1e-10:  # 0으로 나누기 방지
                self._debug_log("  y_vector 정규화 불가 (길이가 0에 가까움), 기본 좌표계 사용")
                return [tuple(kp) for kp in keypoints]  # 변환 없이 반환
            y_vector /= y_norm

            # 회전 행렬 구성
            rotation_matrix = np.column_stack((x_vector, y_vector, z_vector))
            # 중간점 계산
            middle_point = ((x1 + x2)/2) @ rotation_matrix

            # 모든 키포인트 변환
            for kp in keypoints:
                # NaN 및 무한값 검사
                if np.isnan(kp).any() or np.isinf(kp).any():
                    self._debug_log(f"  키포인트에 NaN 또는 무한값 감지: {kp}")
                    converted_keypoints.append((0.0, 0.0, 0.0))
                else:
                    try:
                        converted_point = tuple((kp @ rotation_matrix) - middle_point)
                        converted_keypoints.append(converted_point)
                    except Exception as e:
                        self._debug_log(f"  키포인트 변환 중 오류: {e}")
                        converted_keypoints.append((0.0, 0.0, 0.0))

            return converted_keypoints

        except Exception as e:
            self._debug_log(f"  좌표계 변환 중 예외 발생: {e}")
            # 오류 발생 시 원본 좌표 반환 (튜플 변환)
            return [tuple(kp) if hasattr(kp, '__iter__') else (0.0, 0.0, 0.0) for kp in keypoints]

    def process(self, keypoints):
        """
        키포인트 처리 메인 함수

        Args:
            keypoints: 처리할 키포인트 리스트

        Returns:
            처리된 키포인트 리스트
        """
        try:
            # 입력 검사
            if keypoints is None:
                self._debug_log("process: keypoints가 None입니다")
                return [(0.0, 0.0, 0.0)] * 23  # 기본값 반환 (23개 키포인트)

            if not hasattr(keypoints, '__len__'):
                self._debug_log(f"process: keypoints가 시퀀스가 아닙니다. 타입: {type(keypoints)}")
                return [(0.0, 0.0, 0.0)] * 23  # 기본값 반환

            if len(keypoints) == 0:
                self._debug_log("process: keypoints가 비어 있습니다")
                return [(0.0, 0.0, 0.0)] * 23  # 기본값 반환

            # 키포인트의 10%만 기록 (로그 크기 감소)
            if self.debug_log_enabled and np.random.random() < 0.1:
                self._debug_log(f"process: 키포인트 길이: {len(keypoints)}")
                if len(keypoints) > 0:
                    self._debug_log(f"process: 첫 번째 키포인트 타입: {type(keypoints[0])}")
                    self._debug_log(f"process: 첫 번째 키포인트 값: {keypoints[0]}")

            # 좌표계 변환
            converted_keypoints = self._coordinate_convert(keypoints)

            # 변환 결과 검사
            if not converted_keypoints or len(converted_keypoints) == 0:
                self._debug_log("process: 변환된 키포인트가 비어 있습니다")
                return [(0.0, 0.0, 0.0)] * 23  # 기본값 반환

            # 정규화 적용 (선택적)
            if self.norm_type is not None:
                preprocessed_keypoints = [tuple(kp) for kp in self._points_normalization(np.array(converted_keypoints))]
            else:
                preprocessed_keypoints = converted_keypoints

            # 결과 검사
            if not preprocessed_keypoints or len(preprocessed_keypoints) == 0:
                self._debug_log("process: 전처리된 키포인트가 비어 있습니다")
                return [(0.0, 0.0, 0.0)] * 23  # 기본값 반환

            return preprocessed_keypoints

        except Exception as e:
            self._debug_log(f"process: 키포인트 처리 중 예외 발생: {e}")
            traceback.print_exc()
            return [(0.0, 0.0, 0.0)] * 23  # 오류 발생 시 기본값 반환
'''
