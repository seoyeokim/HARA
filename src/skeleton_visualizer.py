# skeleton_visualizer.py
import cv2
import mediapipe as mp
import numpy as np

class SkeletonVisualizer:
    def __init__(self):
        # MediaPipe 연결 정의 (POSE_CONNECTIONS 사용)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # 스켈레톤 라인 색상 및 두께 설정
        self.connection_color = (0, 255, 0)  # 녹색
        self.landmark_color = (255, 0, 0)    # 빨간색
        self.filtered_landmark_color = (0, 0, 255)  # 파란색
        self.line_thickness = 1
        self.circle_radius = 4

        # 미리 연결 정보 저장
        self.pose_connections = self._get_pose_connections()

    def draw_2d_skeleton(self, frame, landmarks, filtered_keypoints=None):
        """
        2D 스켈레톤 그리기
        Args:
            frame: 시각화할 프레임
            landmarks: MediaPipe 랜드마크
            filtered_keypoints: 필터링된 키포인트 (선택 사항)
        """
        if not landmarks:
            return frame

        height, width, _ = frame.shape

        # MediaPipe 포즈 랜드마크 연결 정보 사용
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=self.landmark_color,
                thickness=1,
                circle_radius=self.circle_radius
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=self.connection_color,
                thickness=self.line_thickness
            )
        )

        # 칼만 필터링된 랜드마크를 위한 스켈레톤 그리기 (있는 경우)
        if filtered_keypoints:
            # 필터링된 랜드마크 그리기
            for i, (x, y) in enumerate(filtered_keypoints):
                cv2.circle(frame, (x, y), self.circle_radius, self.filtered_landmark_color, -1)

            # 필터링된 랜드마크 연결선 그리기
            for connection in self.pose_connections:
                start_idx, end_idx = connection

                if start_idx < len(filtered_keypoints) and end_idx < len(filtered_keypoints):
                    start_point = filtered_keypoints[start_idx]
                    end_point = filtered_keypoints[end_idx]

                    cv2.line(frame, start_point, end_point,
                             self.filtered_landmark_color, self.line_thickness // 2)

        return frame

    def draw_3d_skeleton(self, frame, landmarks_3d, filtered_keypoints_3d=None):
        """
        3D 스켈레톤 그리기 (2D 프로젝션 + 깊이 정보)
        Args:
            frame: 시각화할 프레임
            landmarks_3d: 3D 랜드마크 리스트 [(x,y,z), ...]
            filtered_keypoints_3d: 필터링된 3D 키포인트 (선택 사항)
        """
        if not landmarks_3d:
            return frame

        height, width, _ = frame.shape

        # 원본 랜드마크 연결선 그리기
        for connection in self.pose_connections:
            start_idx, end_idx = connection

            if start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d):
                start_point = (int(landmarks_3d[start_idx][0]), int(landmarks_3d[start_idx][1]))
                end_point = (int(landmarks_3d[end_idx][0]), int(landmarks_3d[end_idx][1]))

                # Z 값에 따라 색상 깊이 표현 (더 가까울수록 밝게)
                start_z = landmarks_3d[start_idx][2]
                end_z = landmarks_3d[end_idx][2]
                avg_z = (start_z + end_z) / 2

                # Z값 범위를 색상으로 매핑 (0~255)
                color_intensity = max(0, min(255, int(255 - abs(avg_z) * 5)))
                z_color = (0, color_intensity, 0)  # 깊이에 따른 녹색 변화

                cv2.line(frame, start_point, end_point, z_color, self.line_thickness)

        # 원본 랜드마크 점 그리기
        for i, (x, y, z) in enumerate(landmarks_3d):
            cv2.circle(frame, (int(x), int(y)), self.circle_radius, self.landmark_color, -1)

            # Z 정보 텍스트 표시 (주요 관절에만)
            if i in [11, 12, 23, 24]:  # 어깨, 골반
                z_text = f"z:{int(z)}"
                cv2.putText(frame, z_text, (int(x) + 5, int(y) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 필터링된 3D 랜드마크 표시 (있는 경우)
        if filtered_keypoints_3d:
            # # 필터링된 랜드마크 연결선 그리기
            # for connection in self.pose_connections:
            #     start_idx, end_idx = connection

            #     if start_idx < len(filtered_keypoints_3d) and end_idx < len(filtered_keypoints_3d):
            #         start_point = (int(filtered_keypoints_3d[start_idx][0]),
            #                       int(filtered_keypoints_3d[start_idx][1]))
            #         end_point = (int(filtered_keypoints_3d[end_idx][0]),
            #                     int(filtered_keypoints_3d[end_idx][1]))

            #         # Z 값에 따른 색상 변화
            #         start_z = filtered_keypoints_3d[start_idx][2]
            #         end_z = filtered_keypoints_3d[end_idx][2]
            #         avg_z = (start_z + end_z) / 2

            #         # Z값 범위를 색상으로 매핑
            #         color_intensity = max(0, min(255, int(255 - abs(avg_z) * 5)))
            #         z_color = (0, 0, color_intensity)  # 깊이에 따른 파란색 변화

            #         cv2.line(frame, start_point, end_point, z_color, self.line_thickness // 2)

            # 필터링된 랜드마크 점 그리기
            for i, (x, y, z) in enumerate(filtered_keypoints_3d):
                cv2.circle(frame, (int(x), int(y)), self.circle_radius,
                          self.filtered_landmark_color, -1)

        return frame

    def _get_pose_connections(self):
        """MediaPipe POSE_CONNECTIONS를 파이썬 튜플 리스트로 변환"""
        # MediaPipe 포즈 연결 정보를 정의
        # 이 연결 정보는 MediaPipe POSE_CONNECTIONS의 구조에 맞게 직접 정의
        connections = [
            # 얼굴
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # 몸통
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            # 다리
            (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
            # 몸통과 얼굴
            (0, 9), (9, 11), (9, 12)
        ]
        return connections

    def draw_direction_arrow(self, frame, com, direction, speed, scale=100):
        """
        이동 방향 화살표 그리기 (카메라 움직임에 덜 민감하게 수정)
        Args:
            frame: 시각화할 프레임
            com: CoM 위치 (x, y) 또는 (x, y, z)
            direction: 방향 벡터 (dx, dy) 또는 (dx, dy, dz)
            speed: 이동 속도
            scale: 화살표 크기 비율
        Returns:
            방향 표시가 추가된 프레임
        """
        if com is None or direction is None:
            return frame

        # 방향 벡터의 시작점 (CoM 위치)
        start_point = (int(com[0]), int(com[1]))

        # 속도 임계값 설정 - 일정 속도 이하는 무시
        speed_threshold = 2.5  # 속도 임계값 (px/s)

        # 속도가 임계값보다 작으면 매우 작은 화살표만 표시하거나 표시하지 않음
        if speed < speed_threshold:
            # 속도가 임계값 이하일 때는 회색으로 표시하고 화살표 길이 최소화
            color = (150, 150, 150)  # 회색
            arrow_length = 15  # 최소 화살표 길이

            # 텍스트에 '속도 표시'
            cv2.putText(frame, f"{speed:.1f} px/s",
                        (start_point[0] + 15, start_point[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # 속도가 임계값보다 크면 정상 화살표 표시
            # 속도에 비선형 매핑 적용 - 작은 움직임에 덜 민감하게
            arrow_length = min(int(20 + (speed - speed_threshold) * 0.7), scale)

            # 화살표 색상 (속도에 따라 변경)
            color = (0, int(min((speed - speed_threshold) * 10, 255)), 255)

            # 속도 텍스트 표시
            cv2.putText(frame, f"{speed:.1f} px/s",
                        (start_point[0] + 15, start_point[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 방향 벡터의 끝점
        end_point = (
            int(start_point[0] + direction[0] * arrow_length),
            int(start_point[1] + direction[1] * arrow_length)
        )

        # 화살표 두께 - 속도에 따라 조정
        thickness = 3 if speed < speed_threshold else 5

        # 화살표 그리기
        cv2.arrowedLine(frame, start_point, end_point, color, thickness,
                       tipLength=0.3 if speed < speed_threshold else 0.4)

        return frame

    def draw_3d_direction(self, frame, com, direction, speed, scale=100):
        """
        3D 이동 방향 시각화 - Z축 표현 각도 조정
        Args:
            frame: 시각화할 프레임
            com: CoM 위치 (x, y, z)
            direction: 방향 벡터 (dx, dy, dz)
            speed: 이동 속도
            scale: 화살표 크기 비율
        Returns:
            방향 표시가 추가된 프레임
        """
        # 2D 화살표 그리기 (기본)
        frame = self.draw_direction_arrow(frame, com, direction[:2], speed, scale)

        # Z축 방향 표시를 위한 준비
        z_direction = direction[2]

        # Z 방향 임계값 - 더 확실한 Z축 움직임이 있을 때만 표시
        z_direction_threshold = 0.6

        # 의미 있는 Z 방향 움직임이 있을 때만 표시
        if abs(z_direction) > z_direction_threshold:
            # CoM 위치 (중심점)
            center_x = int(com[0])
            center_y = int(com[1])

            # 화살표 방향각도 및 길이 설정
            arrow_length = int(25 + abs(z_direction) * 15)  # 화살표 길이

            # Z 방향에 따라 각도와 색상 결정 (정확히 반대 방향으로)
            if z_direction > 0:  # 카메라에서 멀어지는 방향
                # 왼쪽 위로 210도 각도 (CoM Closer의 정확히 반대 방향)
                angle = 210  # 각도 (도 단위)
                arrow_color = (255, 0, 0)  # 파란색

            else:  # 카메라로 다가오는 방향
                # 오른쪽 아래로 30도 각도
                angle = 30  # 각도 (도 단위, 양수는 시계 방향)
                arrow_color = (0, 255, 0)  # 녹색

            # 각도를 라디안으로 변환
            angle_rad = angle * np.pi / 180

            # 화살표 끝점 계산
            end_x = int(center_x + arrow_length * np.cos(angle_rad))
            end_y = int(center_y + arrow_length * np.sin(angle_rad))

            # 좀 더 두껍고 명확한 화살표 그리기
            cv2.arrowedLine(frame,
                        (center_x, center_y),
                        (end_x, end_y),
                        arrow_color,
                        2,  # 두께
                        tipLength=0.3)  # 화살표 팁 길이 비율

            # 좀 더 큰 원으로 시작점 강조 (3D 느낌 강화)
            cv2.circle(frame, (center_x, center_y), 4, arrow_color, -1)

        return frame
