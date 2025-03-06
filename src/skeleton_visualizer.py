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
        self.line_thickness = 2
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
            if i in [0, 11, 12, 23, 24]:  # 코, 어깨, 골반
                z_text = f"z:{int(z)}"
                cv2.putText(frame, z_text, (int(x) + 5, int(y) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 필터링된 3D 랜드마크 표시 (있는 경우)
        if filtered_keypoints_3d:
            # 필터링된 랜드마크 연결선 그리기
            for connection in self.pose_connections:
                start_idx, end_idx = connection

                if start_idx < len(filtered_keypoints_3d) and end_idx < len(filtered_keypoints_3d):
                    start_point = (int(filtered_keypoints_3d[start_idx][0]),
                                  int(filtered_keypoints_3d[start_idx][1]))
                    end_point = (int(filtered_keypoints_3d[end_idx][0]),
                                int(filtered_keypoints_3d[end_idx][1]))

                    # Z 값에 따른 색상 변화
                    start_z = filtered_keypoints_3d[start_idx][2]
                    end_z = filtered_keypoints_3d[end_idx][2]
                    avg_z = (start_z + end_z) / 2

                    # Z값 범위를 색상으로 매핑
                    color_intensity = max(0, min(255, int(255 - abs(avg_z) * 5)))
                    z_color = (0, 0, color_intensity)  # 깊이에 따른 파란색 변화

                    cv2.line(frame, start_point, end_point, z_color, self.line_thickness // 2)

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
