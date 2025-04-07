# com_calculator.py
import numpy as np

class COMCalculator:
    def __init__(self):
        # BSP(Body Segment Parameter) 정의
        # Winter의 연구 기반 분절 질량 비율 (총 질량 대비 각 분절의 비율)
        self.segment_mass_ratios = {
            'head': 0.0694,        # 머리
            'trunk': 0.4346,       # 몸통 (가슴, 복부, 골반 포함)
            'upper_arm': 0.0271,   # 상완 (양쪽 평균)
            'forearm': 0.0162,     # 전완 (양쪽 평균)
            'hand': 0.0061,        # 손 (양쪽 평균)
            'thigh': 0.1416,       # 대퇴 (양쪽 평균)
            'shank': 0.0433,       # 하퇴 (양쪽 평균)
            'foot': 0.0137         # 발 (양쪽 평균)
        }


        # MediaPipe 키포인트 인덱스와 신체 분절 매핑
        self.segment_keypoints = {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 머리 및 얼굴 관련 키포인트
            'trunk': [11, 12, 23, 24],                   # 어깨와 엉덩이 관련 키포인트
            'left_upper_arm': [11, 13],                  # 왼쪽 어깨에서 왼쪽 팔꿈치
            'right_upper_arm': [12, 14],                 # 오른쪽 어깨에서 오른쪽 팔꿈치
            'left_forearm': [13, 15],                    # 왼쪽 팔꿈치에서 왼쪽 손목
            'right_forearm': [14, 16],                   # 오른쪽 팔꿈치에서 오른쪽 손목
            'left_hand': [15, 17, 19, 21],               # 왼쪽 손목 및 손가락
            'right_hand': [16, 18, 20, 22],              # 오른쪽 손목 및 손가락
            'left_thigh': [23, 25],                      # 왼쪽 엉덩이에서 왼쪽 무릎
            'right_thigh': [24, 26],                     # 오른쪽 엉덩이에서 오른쪽 무릎
            'left_shank': [25, 27],                      # 왼쪽 무릎에서 왼쪽 발목
            'right_shank': [26, 28],                     # 오른쪽 무릎에서 오른쪽 발목
            'left_foot': [27, 29, 31],                   # 왼쪽 발목 및 발
            'right_foot': [28, 30, 32]                   # 오른쪽 발목 및 발
        }

        # 분절의 CoM 위치 비율 (분절 길이 대비, 근위에서 원위 방향)
        self.segment_com_ratio = {
            'head': 0.5,             # 머리
            'trunk': 0.5,            # 몸통
            'upper_arm': 0.436,      # 상완
            'forearm': 0.430,        # 전완
            'hand': 0.506,           # 손
            'thigh': 0.433,          # 대퇴
            'shank': 0.433,          # 하퇴
            'foot': 0.5              # 발
        }

    def calculate_segment_com(self, keypoints, segment_name):
        """
        특정 분절의 CoM 계산
        Args:
            keypoints (list): 필터링된 키포인트 좌표 리스트
            segment_name (str): 분절 이름
        Returns:
            tuple: 분절 CoM 좌표 (x, y, z) 또는 (x, y)
        """
        # 분절에 해당하는 키포인트 인덱스 가져오기
        indices = self.segment_keypoints.get(segment_name, [])

        if not indices:
            return None

        # 양쪽 분절 대칭 처리 (좌우 평균값 사용)
        is_symmetric = segment_name in ['upper_arm', 'forearm', 'hand', 'thigh', 'shank', 'foot']

        # 분절 CoM 계산
        if is_symmetric:
            # 좌우 대칭 분절의 경우, 대칭 분절 이름 찾기
            side = segment_name.split('_')[0]  # 'left' 또는 'right'
            other_side = 'right' if side == 'left' else 'left'
            other_segment = segment_name.replace(side, other_side)

            # 분절 양 끝점 (근위, 원위) 찾기
            prox_idx, dist_idx = indices[0], indices[-1]

            # CoM 계산 (선형 보간)
            prox_point = np.array(keypoints[prox_idx])
            dist_point = np.array(keypoints[dist_idx])
            com_ratio = self.segment_com_ratio.get(segment_name.split('_')[-1], 0.5)

            com = prox_point + com_ratio * (dist_point - prox_point)
            return tuple(com)
        else:
            # 단일 분절인 경우 (머리, 몸통)
            segment_points = [np.array(keypoints[idx]) for idx in indices]
            com = np.mean(segment_points, axis=0)
            return tuple(com)

    def calculate_whole_body_com(self, keypoints, include_z=False):
        """
        전신 CoM 계산
        Args:
            keypoints (list): 필터링된 키포인트 좌표 리스트 [(x,y,z), ...] 또는 [(x,y), ...]
            include_z (bool): z좌표를 포함할지 여부
        Returns:
            tuple: 전신 CoM 좌표 (x, y, z) 또는 (x, y)
        """
        # 각 분절의 CoM과 질량비 계산
        segment_coms = {}

        # 머리와 몸통
        segment_coms['head'] = self.calculate_segment_com(keypoints, 'head')
        segment_coms['trunk'] = self.calculate_segment_com(keypoints, 'trunk')

        # 양팔 (좌/우 상완, 전완, 손)
        segment_coms['left_upper_arm'] = self.calculate_segment_com(keypoints, 'left_upper_arm')
        segment_coms['right_upper_arm'] = self.calculate_segment_com(keypoints, 'right_upper_arm')
        segment_coms['left_forearm'] = self.calculate_segment_com(keypoints, 'left_forearm')
        segment_coms['right_forearm'] = self.calculate_segment_com(keypoints, 'right_forearm')
        segment_coms['left_hand'] = self.calculate_segment_com(keypoints, 'left_hand')
        segment_coms['right_hand'] = self.calculate_segment_com(keypoints, 'right_hand')

        # 양다리 (좌/우 대퇴, 하퇴, 발)
        segment_coms['left_thigh'] = self.calculate_segment_com(keypoints, 'left_thigh')
        segment_coms['right_thigh'] = self.calculate_segment_com(keypoints, 'right_thigh')
        segment_coms['left_shank'] = self.calculate_segment_com(keypoints, 'left_shank')
        segment_coms['right_shank'] = self.calculate_segment_com(keypoints, 'right_shank')
        segment_coms['left_foot'] = self.calculate_segment_com(keypoints, 'left_foot')
        segment_coms['right_foot'] = self.calculate_segment_com(keypoints, 'right_foot')

        # BSP 질량비를 적용한 가중 평균으로 전신 CoM 계산
        total_x, total_y, total_z = 0, 0, 0
        valid_mass_sum = 0

        for segment, com in segment_coms.items():
            if com is None:
                continue

            base_segment = segment.split('_')[-1]  # 기본 분절 이름 (side 제거)
            mass_ratio = self.segment_mass_ratios.get(base_segment, 0)

            # 좌/우 대칭 분절은 질량비 절반씩 적용
            if segment.startswith(('left_', 'right_')):
                mass_ratio /= 2

            valid_mass_sum += mass_ratio

            if include_z and len(com) == 3:
                total_x += com[0] * mass_ratio
                total_y += com[1] * mass_ratio
                total_z += com[2] * mass_ratio
            else:
                total_x += com[0] * mass_ratio
                total_y += com[1] * mass_ratio

        # 유효한 질량비로 정규화
        if valid_mass_sum > 0:
            total_x /= valid_mass_sum
            total_y /= valid_mass_sum
            if include_z:
                total_z /= valid_mass_sum
                return (total_x, total_y, total_z)
            else:
                return (total_x, total_y)
        else:
            return None

    def calculate_movement_direction(self, com_history, window_size=5):
        """
        CoM 궤적에서 이동 방향 계산 (3D 지원)
        Args:
        com_history (list): CoM 좌표 리스트 [(x,y,z), ...] 또는 [(x,y), ...]
        window_size (int): 방향 계산에 사용할 최근 프레임 수
        Returns:
        tuple: (direction_vector, speed) - direction_vector는 (dx, dy) 또는 (dx, dy, dz)
        """
        if len(com_history) < window_size:
            # 데이터가 부족하면 3D 또는 2D에 따라 다른 기본값 반환
            is_3d = len(com_history[0]) == 3 if com_history else True
            return ((0, 0, 0) if is_3d else (0, 0)), 0

        # 최근 프레임 사용 - 윈도우 크기 증가
        window_size = min(window_size, len(com_history))
        recent_positions = com_history[-window_size:]

        # 더 안정적인 방향 계산을 위해 이동 평균 적용
        direction_vectors = []
        speeds = []

        # 연속된 프레임들 간의 변화 계산 (프레임 간 변화 백터 수집)
        for i in range(len(recent_positions) - 1):
            start = np.array(recent_positions[i])
            end = np.array(recent_positions[i + 1])
            vector = end - start

            # 프레임 간 거리(속도) 계산
            frame_speed = np.linalg.norm(vector)

            # 노이즈 필터링 - 매우 작은 변화는 무시
            if frame_speed > 1.0:  # 최소 변화 임계값 증가 (0.5에서 1.0으로)
                direction_vectors.append(vector)
                speeds.append(frame_speed)

        # 방향 벡터와 속도 계산
        if direction_vectors:
            # 전체 궤적의 이동 방향 (처음과 끝 비교)
            start_pos = np.array(recent_positions[0])
            end_pos = np.array(recent_positions[-1])
            main_vector = end_pos - start_pos
            main_speed = np.linalg.norm(main_vector) / window_size  # 평균 속도로 변환

            # 프레임 간 변화의 평균 방향과 궤적의 주 방향을 결합
            # 주 방향에 더 높은 가중치 부여 (0.8로 증가)
            if main_speed > 1.0:  # 최소 주 속도 임계값 증가
                avg_vector = np.mean(direction_vectors, axis=0) if direction_vectors else main_vector
                combined_vector = 0.8 * main_vector + 0.2 * avg_vector  # 주 방향에 가중치 증가

                # 평균 속도 계산 (프레임 간 속도와 주 방향 속도의 가중 평균)
                avg_frame_speed = np.mean(speeds) if speeds else 0
                speed = 0.8 * main_speed + 0.2 * avg_frame_speed  # 주 방향 속도에 가중치 증가

                # 방향 벡터 정규화
                combined_speed = np.linalg.norm(combined_vector)
                if combined_speed > 0.5:  # 임계값 증가
                    normalized_direction = combined_vector / combined_speed
                    return tuple(normalized_direction), speed

        # 유효한 방향을 계산할 수 없는 경우 기본값 반환
        is_3d = len(com_history[0]) == 3 if com_history else True
        return ((0, 0, 0) if is_3d else (0, 0)), 0