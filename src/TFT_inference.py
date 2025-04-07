import cv2
import traceback
import numpy as np
import os
import torch
import pandas as pd
from pathlib import Path
import re
from collections import deque
import time
import csv

from pose_estimator import PoseEstimator3D
from kalman_filter import KalmanFilterTracker3D, KeypointPreprocess
from skeleton_visualizer import SkeletonVisualizer
from com_calculator import COMCalculator
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics.base_metrics import Metric as PFMetric

# MPS 사용 가능 여부 확인
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
if mps_available:
    print("MacOS MPS 가속 사용 가능 (Metal Performance Shaders)")
else:
    print("MacOS MPS 가속 사용 불가능, 대체 장치를 사용합니다")

# FocalLoss 클래스 정의 - 모델 로드를 위해 필요 (간소화된 버전)
class FocalLoss(PFMetric):
    """
    추론 전용 껍데기 FocalLoss 클래스
    실제 로직은 필요 없음 (모델 로드를 위한 클래스 정의만 필요)
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        reduction: str = "mean",
        ignore_index: int = -100,
        prediction_length: int = 1,
        **kwargs
    ):
        super().__init__(reduction=reduction, **kwargs)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.prediction_length = prediction_length

        # 클래스 불균형 처리를 위한 알파 설정
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def loss(self, y_pred, y_actual):
        # 추론 시에는 호출되지 않으므로 간단한 더미 반환
        return torch.tensor(0.0, device=y_pred.device)

    # 기본 인터페이스 메서드는 유지
    def to_prediction(self, y_actual):
        return y_actual

    def to_quantiles(self, y_pred):
        return y_pred

    def update(self, y_pred, y_actual):
        self.val = torch.tensor(0.0)

    def compute(self):
        return self.val

# BCE Loss 클래스 정의 (2클래스 모델용)
class BCEWithLogitsLoss(PFMetric):
    """
    Binary Cross Entropy Loss for binary classification in time series forecasting.
    Compatible with pytorch_forecasting metric system.
    """
    def __init__(
        self,
        reduction: str = "mean",
        weight: torch.Tensor = None,
        pos_weight: torch.Tensor = None,
        prediction_length: int = 1,
        **kwargs
    ):
        super().__init__(reduction=reduction, **kwargs)
        self.prediction_length = prediction_length

        # 클래스 가중치 설정
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

        # 양성 클래스 가중치 설정
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def loss(self, y_pred, y_actual):
        # 추론 시에는 호출되지 않으므로 간단한 더미 반환
        return torch.tensor(0.0, device=y_pred.device)

    # 기본 인터페이스 메서드는 유지
    def to_prediction(self, y_actual):
        return y_actual

    def to_quantiles(self, y_pred):
        return y_pred

    def update(self, y_pred, y_actual):
        self.val = torch.tensor(0.0)

    def compute(self):
        return self.val

# 커스텀 TFT 클래스 정의 (시각화 오류 우회)
class CustomTFT(TemporalFusionTransformer):
    def log_prediction(self, x, out, batch_idx, **kwargs):
        # 시각화 없이 빈 로그 반환
        return {}

class PedestrianBehaviorPredictor:
    def __init__(self, model_path=None, checkpoint_dir=None, max_encoder_length=20, use_mps=True, binary_mode=False):
        """
        보행자 행동 예측기 초기화

        Args:
            model_path: 모델 체크포인트 파일 경로. None인 경우 checkpoint_dir에서 가장 최신 체크포인트 사용
            checkpoint_dir: 체크포인트 파일이 저장된 디렉터리 경로
            max_encoder_length: 인코더의 최대 시퀀스 길이 (과거 프레임 수)
            use_mps: macOS에서 MPS(Metal Performance Shaders) 가속 사용 여부
            binary_mode: 2클래스 모드 사용 여부 (추가됨)
        """
        # MacOS에서 MPS 사용 가능 여부 확인
        if use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("MPS 가속 사용 중: Metal Performance Shaders")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("CUDA 가속 사용 중")
        else:
            self.device = torch.device('cpu')
            print("CPU 사용 중")

        self.max_encoder_length = max_encoder_length
        self.binary_mode = binary_mode  # binary_mode 속성 설정

        # 클래스 이름 설정 (2클래스 또는 4클래스)
        if binary_mode:
            self.class_names = ['Standing', 'Walking']
        else:
            self.class_names = ['Standing', 'Start Walking', 'Walking', 'Finish Walking']

        # 모델 경로가 제공되지 않은 경우, 가장 최신 체크포인트 찾기
        if model_path is None and checkpoint_dir is not None:
            # 바이너리 모드에 맞는 체크포인트 파일 패턴 찾기
            if binary_mode:
                pattern = "tft-pedestrian-binary-*.ckpt"
            else:
                pattern = "tft-pedestrian-epoch=*.ckpt"

            try:
                model_path = self.find_latest_checkpoint(checkpoint_dir, pattern)
            except FileNotFoundError:
                # 패턴이 일치하는 파일이 없으면 다른 패턴으로 시도
                if binary_mode:
                    # 다른 바이너리 모델 패턴 시도
                    alternative_patterns = ["*binary*.ckpt", "*.ckpt"]
                else:
                    # 다른 4클래스 모델 패턴 시도
                    alternative_patterns = ["*epoch*.ckpt", "*.ckpt"]

                for alt_pattern in alternative_patterns:
                    try:
                        print(f"패턴 '{pattern}'으로 체크포인트를 찾을 수 없어 '{alt_pattern}' 패턴으로 시도합니다.")
                        model_path = self.find_latest_checkpoint(checkpoint_dir, alt_pattern)
                        break
                    except FileNotFoundError:
                        continue

                if model_path is None:
                    raise FileNotFoundError(f"디렉터리에서 체크포인트 파일을 찾을 수 없습니다: {checkpoint_dir}")

        if model_path:
            self.load_model(model_path)
        else:
            raise ValueError("model_path 또는 checkpoint_dir 중 하나는 제공되어야 합니다.")

        # 보행자 ID 인코더 초기화 (인퍼런스 시에는 값이 중요하지 않음)
        # 넘파이 배열로 변환하여 fit
        self.pedestrian_encoder = NaNLabelEncoder(add_nan=True)
        self.pedestrian_encoder.fit(np.array(["dummy_id", "inference_id"]))

        # 프레임 버퍼 초기화 (시계열 데이터 저장)
        self.frame_buffer = []
        self.current_frame = 0

        # 디버깅 로그 추가
        self.debug_log_enabled = True

    def debug_log(self, message):
        """디버깅 로그 출력 함수"""
        if self.debug_log_enabled:
            print(message)

    def find_latest_checkpoint(self, checkpoint_dir, pattern="tft-pedestrian-*.ckpt"):
        """
        체크포인트 디렉터리에서 패턴에 맞는 가장 최신 체크포인트 파일을 찾습니다.
        """
        # 디렉터리 내 파일 목록 가져오기
        checkpoint_files = list(Path(checkpoint_dir).glob(pattern))

        if not checkpoint_files:
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_dir} (패턴: {pattern})")

        # 에폭 번호 추출 함수
        def extract_number(filepath):
            # 파일 이름에서 숫자 패턴 찾기
            str_path = str(filepath)

            # 'epoch=' 패턴 검색
            epoch_match = re.search(r'epoch=(\d+)', str_path)
            if epoch_match:
                return int(epoch_match.group(1))

            # 'binary-' 패턴 검색
            binary_match = re.search(r'binary-(\d+)', str_path)
            if binary_match:
                return int(binary_match.group(1))

            # 일반적인 숫자 패턴 검색
            num_match = re.search(r'(\d+)', str_path)
            if num_match:
                return int(num_match.group(1))

            # 매치되지 않을 경우
            return 0

        try:
            # 에폭 번호에 따라 체크포인트 파일 정렬 및 가장 큰 번호 선택
            latest_checkpoint = max(checkpoint_files, key=extract_number)
            print(f"가장 최신 체크포인트 파일: {latest_checkpoint}")
            return str(latest_checkpoint)
        except Exception as e:
            print(f"체크포인트 파일 선택 중 오류 발생: {e}")
            # 오류 발생 시 첫 번째 파일 반환
            if checkpoint_files:
                print(f"첫 번째 체크포인트 파일 사용: {checkpoint_files[0]}")
                return str(checkpoint_files[0])
            raise

    def load_model(self, model_path):
        """
        저장된 모델 가중치를 로드합니다.
        """
        try:
            # 적절한 장치로 모델 로드를 위한 map_location 설정
            if self.device.type == 'mps':
                # MPS로 저장된 모델은 MPS로, CUDA로 저장된 모델은 MPS로 변환
                map_location = {'cuda:0': 'mps', 'mps:0': 'mps'}
            elif self.device.type == 'cuda':
                # 기존 방식대로 CUDA 사용
                map_location = self.device
            else:
                # CPU로 변환
                map_location = {'cuda:0': 'cpu', 'mps:0': 'cpu'}

            # 모델 로드 시도
            try:
                # 중요: FocalLoss 클래스를 globals()에 등록
                import sys
                sys.modules['__main__'].FocalLoss = FocalLoss
                sys.modules['__main__'].BCEWithLogitsLoss = BCEWithLogitsLoss

                self.model = CustomTFT.load_from_checkpoint(
                    model_path,
                    map_location=map_location
                )
            except Exception as e:
                print(f"첫 번째 로드 방법 실패: {e}")
                # 두 번째 로드 방법: strict=False 설정하고 손실 함수 명시적 전달
                if self.binary_mode:
                    loss_function = BCEWithLogitsLoss()
                else:
                    loss_function = FocalLoss()

                self.model = CustomTFT.load_from_checkpoint(
                    model_path,
                    map_location=map_location,
                    loss=loss_function,
                    strict=False  # 추가: 일부 가중치 불일치 허용
                )

            self.model.to(self.device)
            self.model.eval()
            print(f"모델을 성공적으로 로드했습니다: {model_path} (장치: {self.device}, 바이너리 모드: {self.binary_mode})")

            # 모델 출력 크기 확인 - output_size 속성이 없는 경우 대체 방법 사용
            try:
                if hasattr(self.model, 'output_size'):
                    output_size = self.model.output_size
                elif hasattr(self.model, 'hparams') and hasattr(self.model.hparams, 'output_size'):
                    output_size = self.model.hparams.output_size
                elif hasattr(self.model, 'output_layer') and hasattr(self.model.output_layer, 'out_features'):
                    output_size = self.model.output_layer.out_features
                else:
                    # 출력 크기를 결정할 수 없는 경우, 바이너리 모드 플래그에 따라 기본값 사용
                    output_size = 2 if self.binary_mode else 4
                    print(f"모델의 출력 크기를 결정할 수 없어 기본값을 사용합니다: {output_size}")

                print(f"모델 출력 크기: {output_size}")

                # 출력 크기로 이진 분류 모드 자동 감지
                if output_size == 2 and not self.binary_mode:
                    print("출력 크기가 2이므로 이진 분류 모드로 전환합니다.")
                    self.binary_mode = True
                    self.class_names = ['Standing', 'Walking']
                elif output_size == 4 and self.binary_mode:
                    print("경고: 이진 모드로 지정했지만 모델 출력이 4클래스입니다. 4클래스 모드로 전환합니다.")
                    self.binary_mode = False
                    self.class_names = ['Standing', 'Start Walking', 'Walking', 'Finish Walking']

                # 모델 테스트 예측 수행 (정상 작동 확인)
                print("모델 테스트 예측 실행 중...")
                try:
                    dummy_input = {
                        'x_cat': torch.zeros((1, 1, 1), dtype=torch.long, device=self.device),
                        'x_real': torch.zeros((1, self.max_encoder_length, 70), dtype=torch.float32, device=self.device),
                        'target': torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device),
                    }

                    with torch.no_grad():
                        self.model.eval()
                        _ = self.model(dummy_input)
                    print("모델 테스트 예측 성공!")
                except Exception as e:
                    print(f"모델 테스트 예측 실패: {e}")
                    traceback.print_exc()

            except Exception as e:
                print(f"모델 출력 크기 확인 중 오류 발생: {e}")
                # 오류가 발생한 경우 바이너리 모드에 따라 기본값 사용
                if self.binary_mode:
                    print("이진 분류 모드를 유지합니다.")
                    self.class_names = ['Standing', 'Walking']
                else:
                    print("4클래스 분류 모드를 유지합니다.")
                    self.class_names = ['Standing', 'Start Walking', 'Walking', 'Finish Walking']

        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            traceback.print_exc()
            raise

    def prepare_data(self, keypoints, frame_id=None):
        """
        처리된 키포인트를 TFT 모델 입력 형식으로 변환

        Args:
            keypoints: 처리된 키포인트 리스트 [(x,y,z), (x,y,z), ...] (23개 튜플)
            frame_id: 현재 프레임 ID (None인 경우 자동으로 증가)

        Returns:
            변환된 데이터프레임
        """
        # 현재 프레임 ID 설정
        if frame_id is None:
            self.current_frame += 1
            frame_id = self.current_frame

        # 기본 데이터 딕셔너리 구성
        data = {
            'frame': frame_id,
            'pedestrian_id': 'inference_id',  # 인퍼런스용 임의의 ID
        }

        # 2클래스/4클래스 모드에 따라 적절한 라벨 키 설정
        if self.binary_mode:
            data['binary_label'] = 0  # 더미 레이블 (인퍼런스 시에는 사용되지 않음)
        else:
            data['behavior_label'] = 0  # 더미 레이블 (인퍼런스 시에는 사용되지 않음)

        # 디버깅: keypoints 기본 정보 로깅
        if self.debug_log_enabled:
            self.debug_log(f"\n[prepare_data] 프레임 {frame_id}의 디버깅 정보:")
            self.debug_log(f"  - 입력 keypoints 타입: {type(keypoints)}")
            self.debug_log(f"  - 입력 keypoints 길이: {len(keypoints) if hasattr(keypoints, '__len__') else 'N/A'}")

        # 키포인트 유효성 검사 및 변환
        try:
            # keypoints가 리스트가 아닌 경우 변환 시도
            if not isinstance(keypoints, (list, tuple, np.ndarray)):
                self.debug_log(f"  - 키포인트가 list, tuple, ndarray가 아님: {type(keypoints)}")
                # 변환 시도
                if hasattr(keypoints, '__iter__'):
                    keypoints = list(keypoints)
                    self.debug_log(f"  - 키포인트를 리스트로 변환: {len(keypoints)} 항목")
                else:
                    raise TypeError(f"키포인트를 처리할 수 없는 형식: {type(keypoints)}")

            # 항목 수 검증
            if len(keypoints) != 23:  # 필요한 키포인트 수 (23개 키포인트)
                self.debug_log(f"  - [경고] 키포인트 개수가 예상(23)과 다릅니다: {len(keypoints)}")
                # 부족한 경우 채우기, 많은 경우 자르기
                if len(keypoints) < 23:
                    # 부족한 키포인트는 0으로 채우기
                    keypoints = list(keypoints) + [(0.0, 0.0, 0.0)] * (23 - len(keypoints))
                else:
                    # 초과하는 경우 앞에서부터 23개만 사용
                    keypoints = keypoints[:23]
                self.debug_log(f"  - 키포인트 개수 조정 후: {len(keypoints)}")

            # 각 키포인트 처리 및 평탄화
            feature_idx = 0
            for i, keypoint in enumerate(keypoints):
                # 유효성 검사: 각 keypoint가 좌표 튜플인지 확인
                if not isinstance(keypoint, (tuple, list, np.ndarray)) or len(keypoint) != 3:
                    self.debug_log(f"  - [경고] 키포인트 {i}의 형식이 올바르지 않습니다: {type(keypoint)}")
                    # 문제가 있는 경우 기본값 (0,0,0) 사용
                    keypoint = (0.0, 0.0, 0.0)

                # x, y, z 좌표 추출하여 특성으로 저장
                for j, coord in enumerate(keypoint):
                    try:
                        # 숫자형으로 변환 시도
                        coord_value = float(coord)
                        # NaN 또는 Infinity 값 확인
                        if not np.isfinite(coord_value):
                            self.debug_log(f"  - [경고] 키포인트 {i}의 좌표 {j}가 유한값이 아님: {coord}")
                            coord_value = 0.0
                    except (ValueError, TypeError):
                        self.debug_log(f"  - [경고] 키포인트 {i}의 좌표 {j}를 float로 변환할 수 없음: {coord}")
                        coord_value = 0.0

                    # 최종 좌표값 저장
                    data[f'kp_{feature_idx}'] = coord_value
                    feature_idx += 1

            # 특성 개수 확인
            self.debug_log(f"  - 생성된 특성 개수: {feature_idx}")
            if feature_idx != 69:  # 23 keypoints * 3 coordinates = 69
                self.debug_log(f"  - [경고] 특성 개수가 예상(69)과 다릅니다: {feature_idx}")
                # 부족한 특성은 0으로 채우기
                for i in range(feature_idx, 69):
                    data[f'kp_{i}'] = 0.0

            # 모든 69개 특성이 존재하는지 최종 확인
            for i in range(69):
                if f'kp_{i}' not in data:
                    self.debug_log(f"  - [경고] 누락된 특성: kp_{i}")
                    data[f'kp_{i}'] = 0.0

        except Exception as e:
            self.debug_log(f"  - [오류] 키포인트 처리 중 예외 발생: {e}")
            traceback.print_exc()
            # 오류가 발생한 경우, 모든 키포인트 특성을 0으로 초기화
            for i in range(69):
                data[f'kp_{i}'] = 0.0

        # 프레임 버퍼에 추가
        self.frame_buffer.append(data)

        # 버퍼 크기가 max_encoder_length를 초과하면 가장 오래된 프레임 제거
        if len(self.frame_buffer) > self.max_encoder_length:
            self.frame_buffer.pop(0)

        # 버퍼의 모든 프레임을 포함하는 데이터프레임 생성
        buffer_df = pd.DataFrame(self.frame_buffer)

        # 데이터 타입 변환
        buffer_df["pedestrian_id"] = buffer_df["pedestrian_id"].astype('category')
        buffer_df["frame"] = buffer_df["frame"].astype(int)

        return buffer_df

    def create_dataset(self, df):
        """
        인퍼런스용 TimeSeriesDataSet 생성
        """
        try:
            # 입력 데이터 (Keypoints + CoM)
            feature_columns = [f"kp_{i}" for i in range(69)]  # 모든 키포인트 + CoM 포인트

            # 데이터프레임 유효성 검사 - 필수 컬럼 확인
            required_cols = feature_columns + ["frame", "pedestrian_id"]
            if self.binary_mode:
                required_cols.append("binary_label")
                target_col = "binary_label"
            else:
                required_cols.append("behavior_label")
                target_col = "behavior_label"

            for col in required_cols:
                if col not in df.columns:
                    # 타겟 컬럼이 없는 경우 임의로 추가
                    if col == "binary_label" or col == "behavior_label":
                        df[col] = 0  # 더미 값 추가
                        self.debug_log(f"  - 누락된 타겟 열 '{col}'을 더미 값으로 추가했습니다.")
                    else:
                        raise ValueError(f"필수 열이 데이터프레임에 없습니다: {col}")

            # TimeSeriesDataSet 생성
            dataset = TimeSeriesDataSet(
                df,
                time_idx="frame",  # 프레임 번호
                target=target_col,  # 예측할 행동 클래스
                group_ids=["pedestrian_id"],  # 보행자별 데이터 그룹화
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=1,  # 1개 클래스 출력
                static_categoricals=["pedestrian_id"],  # 보행자 ID를 Static Variable로 설정
                time_varying_known_reals=["frame"],  # 시간에 따라 변하는 변수
                time_varying_unknown_reals=feature_columns,  # 입력 데이터
                categorical_encoders={"pedestrian_id": self.pedestrian_encoder},  # 인코더 명시적 지정
                target_normalizer=None,  # 분류 문제이므로 normalizer 불필요
                allow_missing_timesteps=True,  # 중간 프레임 누락 허용
                min_encoder_length=1,  # 최소 인코더 길이 감소
            )

            return dataset

        except Exception as e:
            self.debug_log(f"[오류] create_dataset 중 예외 발생: {e}")
            traceback.print_exc()
            raise

    def predict(self, frame_id=None):
        """
        프레임 버퍼에 저장된 데이터를 기반으로 보행자 행동 예측

        Args:
            frame_id: 현재 프레임 ID (사용되지 않음)

        Returns:
            예측된 행동 클래스 인덱스, 클래스 이름, 각 클래스에 대한 확률
        """
        # 성능 측정 시작
        predict_start_time = time.time()

        # 디버깅 메시지 추가
        self.debug_log(f"predict 메서드 호출됨, 프레임 버퍼 크기: {len(self.frame_buffer)}")

        # 충분한 프레임이 쌓이지 않았으면 예측 불가
        if len(self.frame_buffer) < self.max_encoder_length:
            # 이진 모드일 경우 확률 분포 조정
            if self.binary_mode:
                return None, f"Collecting data... ({len(self.frame_buffer)}/{self.max_encoder_length})", [0.5, 0.5]
            else:
                return None, f"Collecting data... ({len(self.frame_buffer)}/{self.max_encoder_length})", [0.25, 0.25, 0.25, 0.25]

        # 단계별 처리 시간 측정을 위한 변수들
        prepare_data_time = 0
        dataloader_time = 0
        model_inference_time = 0

        # 이미 프레임 버퍼가 존재함을 가정
        try:
            # 1. 데이터 준비 시간 측정
            data_prep_start = time.time()
            df = pd.DataFrame(self.frame_buffer)

            # 데이터프레임 유효성 검사
            self.debug_log("데이터프레임 형태 검증:")
            self.debug_log(f"  - 행 수: {len(df)}")
            self.debug_log(f"  - 열 수: {len(df.columns)}")
            self.debug_log(f"  - 열 목록: {df.columns.tolist()[:10]}...")  # 처음 10개 열만 출력

            # NaN 값 확인
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                self.debug_log(f"  - NaN 값이 있는 열: {nan_cols}")
                # NaN 값을 0으로 대체
                df = df.fillna(0)
                self.debug_log("  - NaN 값을 0으로 대체했습니다.")

            # 데이터셋 생성
            dataset = self.create_dataset(df)
            data_prep_end = time.time()
            prepare_data_time = data_prep_end - data_prep_start

            # 2. 데이터로더 생성 시간 측정
            dataloader_start = time.time()
            dataloader = dataset.to_dataloader(
                train=False,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            dataloader_end = time.time()
            dataloader_time = dataloader_end - dataloader_start

            # 예측 수행
            with torch.no_grad():
                # batch 가져오기
                batch_start = time.time()
                batch = next(iter(dataloader))
                batch_end = time.time()
                batch_time = batch_end - batch_start

                # batch 타입 확인 및 디버깅
                self.debug_log("예측 배치 정보:")
                if isinstance(batch, dict):
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            self.debug_log(f"  - {key}: 텐서 크기 {value.shape}, 타입 {value.dtype}")
                        else:
                            self.debug_log(f"  - {key}: {type(value)}")
                elif isinstance(batch, tuple) or isinstance(batch, list):
                    # 튜플이나 리스트인 경우, 첫 번째 요소를 batch로 사용
                    self.debug_log(f"  - batch가 튜플/리스트입니다. 첫 번째 요소를 사용합니다.")
                    if len(batch) > 0:
                        batch = batch[0]  # 첫 번째 요소 사용
                        if isinstance(batch, dict):
                            self.debug_log(f"  - 첫 번째 요소는 딕셔너리입니다.")
                        else:
                            self.debug_log(f"  - 첫 번째 요소 타입: {type(batch)}")
                    else:
                        raise ValueError("빈 배치가 반환되었습니다.")
                else:
                    self.debug_log(f"  - 예상치 못한 배치 타입: {type(batch)}")
                    # 기본값 반환
                    if self.binary_mode:
                        return 0, "Standing", [0.8, 0.2]
                    else:
                        return 0, "Standing", [0.6, 0.2, 0.1, 0.1]

                # 모델 예측 준비 - 현재 장치 사용
                try:
                    # 3. 모델 추론 시간 측정
                    inference_start = time.time()

                    # 현재 장치 확인
                    current_device = self.device
                    self.debug_log(f"현재 모델 장치: {current_device}")

                    # 모든 텐서가 올바른 장치에 있는지 확인
                    if isinstance(batch, dict):
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                # 모든 텐서를 현재 모델 장치로 명시적으로 이동
                                batch[key] = value.to(current_device)
                                self.debug_log(f"  - {key} 텐서를 {current_device}로 이동함")

                    # 모델이 추론 모드인지 확인
                    self.model.eval()

                    # 메모리 관리를 위한 초기화
                    if current_device.type == 'mps':
                        # MPS 장치의 캐시 비우기
                        torch.mps.empty_cache()
                        self.debug_log("MPS 메모리 캐시 비움")

                    # 장치 동기화 보장
                    if current_device.type == 'mps':
                        torch.mps.synchronize()
                        self.debug_log("MPS 장치 동기화 완료")

                    # 모델 예측 실행
                    self.debug_log("모델 예측 시작")
                    outputs = self.model(batch)

                    # MPS 장치 동기화
                    if current_device.type == 'mps':
                        torch.mps.synchronize()

                    inference_end = time.time()
                    model_inference_time = inference_end - inference_start

                    self.debug_log(f"모델 예측 완료 (시간: {model_inference_time*1000:.1f}ms)")

                    # 4. 후처리 시간 측정
                    postproc_start = time.time()

                    # 출력 확인
                    self.debug_log(f"모델 출력 타입: {type(outputs)}")
                    if hasattr(outputs, '__dir__'):
                        self.debug_log(f"출력 속성: {dir(outputs)[:10]}")

                    # 테스트 데이터셋 평가 코드에서 사용한 방식으로 확률 추출
                    if hasattr(outputs, 'prediction'):
                        # 예측값 처리
                        predictions = outputs.prediction

                        # 예측값 처리 - 이진 분류/다중 분류에 따른 확률 계산 방식 차이
                        if self.binary_mode:
                            # 이진 분류의 경우 - 출력 크기 확인
                            if predictions.size(-1) == 2:
                                # 두 클래스에 대한 로짓이 있는 경우 (Softmax로 확률 변환)
                                probabilities = torch.softmax(predictions, dim=-1)
                                predicted_class = torch.argmax(probabilities, dim=-1)
                            else:
                                # 단일 로짓인 경우 (Sigmoid로 확률 변환)
                                prob = torch.sigmoid(predictions)
                                predicted_class = (prob > 0.5).long()
                                # [1-p, p] 형식으로 변환
                                probabilities = torch.cat([1-prob, prob], dim=-1)
                        else:
                            # 다중 클래스의 경우 항상 softmax 사용
                            probabilities = torch.softmax(predictions, dim=-1)
                            predicted_class = torch.argmax(probabilities, dim=-1)

                        self.debug_log(f"확률 텐서 크기: {probabilities.shape}")

                        # 마지막 타임스텝의 예측 추출
                        if probabilities.dim() >= 3:
                            probabilities = probabilities[0, -1, :]
                            predicted_class = predicted_class[0, -1]
                        else:
                            self.debug_log(f"예상치 못한 확률 차원: {probabilities.shape}")
                            probabilities = probabilities.squeeze()
                            predicted_class = predicted_class.squeeze()

                        # 확률과 예측 클래스를 numpy 배열로 변환
                        probabilities = probabilities.cpu().numpy()
                        predicted_class = predicted_class.cpu().numpy()

                        # 스칼라로 변환
                        if isinstance(predicted_class, np.ndarray):
                            predicted_class = predicted_class.item()

                        predicted_class_name = self.class_names[predicted_class]

                        postproc_end = time.time()
                        postproc_time = postproc_end - postproc_start

                        # 전체 예측 시간
                        predict_end_time = time.time()
                        total_predict_time = predict_end_time - predict_start_time

                        # 성능 로그 출력
                        self.debug_log("\n=== 예측 성능 상세 ===")
                        self.debug_log(f"데이터 준비 시간: {prepare_data_time*1000:.1f}ms ({prepare_data_time/total_predict_time*100:.1f}%)")
                        self.debug_log(f"데이터로더 생성 시간: {dataloader_time*1000:.1f}ms ({dataloader_time/total_predict_time*100:.1f}%)")
                        self.debug_log(f"배치 가져오기 시간: {batch_time*1000:.1f}ms ({batch_time/total_predict_time*100:.1f}%)")
                        self.debug_log(f"모델 추론 시간: {model_inference_time*1000:.1f}ms ({model_inference_time/total_predict_time*100:.1f}%)")
                        self.debug_log(f"후처리 시간: {postproc_time*1000:.1f}ms ({postproc_time/total_predict_time*100:.1f}%)")
                        self.debug_log(f"총 예측 시간: {total_predict_time*1000:.1f}ms")
                        # FPS 정보 추가
                        fps = 1.0 / total_predict_time if total_predict_time > 0 else 0
                        self.debug_log(f"추론 FPS: {fps:.1f}")
                        self.debug_log("=====================")

                        return predicted_class, predicted_class_name, probabilities
                    else:
                        self.debug_log("모델 출력에 prediction 속성이 없습니다")
                        # 기본값 반환
                        if self.binary_mode:
                            return 0, "Standing", [0.8, 0.2]
                        else:
                            return 0, "Standing", [0.6, 0.2, 0.1, 0.1]

                except Exception as e:
                    self.debug_log(f"모델 예측 중 오류 발생: {e}")
                    traceback.print_exc()

                    # MPS 특정 오류 처리
                    if "MPS" in str(e) or "Placeholder storage" in str(e):
                        self.debug_log("MPS 관련 오류 감지됨, 기본값 반환")

                    # 기본값 반환
                    if self.binary_mode:
                        return 0, "Standing", [0.8, 0.2]
                    else:
                        return 0, "Standing", [0.6, 0.2, 0.1, 0.1]

        except Exception as e:
            self.debug_log(f"예측 과정 중 오류 발생: {e}")
            traceback.print_exc()

            # 기본값 반환 (바이너리 모드에 따라 다르게 반환)
            if self.binary_mode:
                return 0, "Standing", [0.8, 0.2]  # 이진 분류 기본값
            else:
                return 0, "Standing", [0.6, 0.2, 0.1, 0.1]  # 4클래스 기본값

    # predict_heuristic 함수 주석 처리
    """
    def predict_heuristic(self):
        '''
        COM 이동 패턴 기반 휴리스틱 행동 예측

        Returns:
            예측 클래스, 클래스 이름, 확률 분포
        '''
        if len(self.frame_buffer) < 10:
            # 바이너리 모드에 따른 기본값 반환
            if self.binary_mode:
                return 0, "Standing", [0.8, 0.2]  # 이진 분류 기본값
            else:
                return 0, "Standing", [0.7, 0.1, 0.1, 0.1]  # 4클래스 기본값

        try:
            # 최근 10프레임의 CoM 데이터 추출
            recent_coms = []
            for i in range(-min(10, len(self.frame_buffer)), 0):
                try:
                    frame_data = self.frame_buffer[i]
                    # kp_66, kp_67, kp_68이 CoM의 x, y, z 좌표
                    if all(f'kp_{j}' in frame_data for j in range(66, 69)):
                        x = frame_data['kp_66']
                        y = frame_data['kp_67']
                        z = frame_data['kp_68']
                        recent_coms.append((x, y, z))
                except (KeyError, IndexError) as e:
                    self.debug_log(f"  [오류] CoM 데이터 추출 중 예외 발생 (인덱스 {i}): {e}")
                    continue

            if len(recent_coms) < 5:
                # 바이너리 모드에 따른 기본값 반환
                if self.binary_mode:
                    return 0, "Standing", [0.8, 0.2]  # 이진 분류 기본값
                else:
                    return 0, "Standing", [0.7, 0.1, 0.1, 0.1]  # 4클래스 기본값

            # 전체 이동 거리 계산
            total_distance = 0
            for i in range(1, len(recent_coms)):
                dx = recent_coms[i][0] - recent_coms[i-1][0]
                dy = recent_coms[i][1] - recent_coms[i-1][1]
                total_distance += np.sqrt(dx*dx + dy*dy)

            # 시작점과 끝점 사이 거리
            start_to_end = np.sqrt(
                (recent_coms[-1][0] - recent_coms[0][0])**2 +
                (recent_coms[-1][1] - recent_coms[0][1])**2
            )

            # 앞쪽 절반과 뒤쪽 절반의 이동 거리 계산
            mid = len(recent_coms) // 2
            first_half_distance = 0
            for i in range(1, mid):
                dx = recent_coms[i][0] - recent_coms[i-1][0]
                dy = recent_coms[i][1] - recent_coms[i-1][1]
                first_half_distance += np.sqrt(dx*dx + dy*dy)

            second_half_distance = 0
            for i in range(mid, len(recent_coms)):
                dx = recent_coms[i][0] - recent_coms[i-1][0]
                dy = recent_coms[i][1] - recent_coms[i-1][1]
                second_half_distance += np.sqrt(dx*dx + dy*dy)

            # 바이너리 모드에 따른 행동 분류
            if self.binary_mode:
                # 이진 분류 모드 (Standing / Walking)
                if total_distance < 20:  # 거의 움직이지 않음
                    return 0, "Standing", [0.9, 0.1]
                else:  # 움직임 감지됨
                    return 1, "Walking", [0.1, 0.9]
            else:
                # 4클래스 모드
                if total_distance < 20:  # 거의 움직이지 않음
                    return 0, "Standing", [0.8, 0.1, 0.05, 0.05]
                elif start_to_end > 50:  # 많이 이동
                    if first_half_distance < second_half_distance * 0.7:  # 가속
                        return 1, "Start Walking", [0.05, 0.8, 0.1, 0.05]
                    elif second_half_distance < first_half_distance * 0.7:  # 감속
                        return 3, "Finish Walking", [0.05, 0.1, 0.05, 0.8]
                    else:  # 일정한 속도
                        return 2, "Walking", [0.05, 0.1, 0.8, 0.05]
                else:  # 약간 움직임
                    if second_half_distance > first_half_distance * 1.5:  # 가속
                        return 1, "Start Walking", [0.1, 0.7, 0.1, 0.1]
                    elif first_half_distance > second_half_distance * 1.5:  # 감속
                        return 3, "Finish Walking", [0.1, 0.1, 0.1, 0.7]
                    else:
                        return 2, "Walking", [0.1, 0.1, 0.7, 0.1]

        except Exception as e:
            self.debug_log(f"휴리스틱 예측 중 예외 발생: {e}")
            traceback.print_exc()
            # 바이너리 모드에 따른 기본값 반환
            if self.binary_mode:
                return 0, "Standing", [0.8, 0.2]  # 이진 분류 기본값
            else:
                return 0, "Standing", [0.6, 0.2, 0.1, 0.1]  # 4클래스 기본값
    """

    def reset(self):
        """
        프레임 버퍼 초기화 (새로운 시퀀스 시작 시 호출)
        """
        self.frame_buffer = []
        self.current_frame = 0
        self.debug_log("프레임 버퍼 초기화 완료")


class TFTInferenceSystem:
    def __init__(self, checkpoint_dir, max_encoder_length=20, use_mps=True, binary_mode=False, csv_export=True, csv_filename='keypoints_export_normalized.csv'):
        """
        TFT 추론 시스템 초기화

        Args:
            checkpoint_dir: TFT 모델 체크포인트가 저장된 디렉터리
            max_encoder_length: TFT 모델 인코더의 최대 시퀀스 길이
            use_mps: MacOS에서 MPS 가속 사용 여부
            binary_mode: 이진 분류 모드 사용 여부
            csv_export: CSV 파일로 키포인트 데이터 내보내기 여부
            csv_filename: 내보낼 CSV 파일 이름
        """
        self.tft_keypoint_processor = KeypointPreprocess(True)  # 정규화 모드 활성화
        self.binary_mode = binary_mode  # binary_mode 속성 추가

        # TFT 예측기 초기화
        self.behavior_predictor = PedestrianBehaviorPredictor(
            checkpoint_dir=checkpoint_dir,
            max_encoder_length=max_encoder_length,
            use_mps=use_mps,
            binary_mode=binary_mode
        )

        # 행동 예측 결과를 위한 버퍼 (스무딩을 위해)
        self.prediction_buffer = deque(maxlen=5)

        # 프레임 카운터
        self.frame_counter = 0

        # 성능 모니터링 변수
        self.perf_monitor = {
            'prediction_times': deque(maxlen=30),  # 최근 30개 예측 시간
        }

        # CSV 내보내기 설정
        self.csv_export = csv_export
        self.csv_filename = csv_filename

        # CSV 파일 초기화 (헤더 작성)
        if self.csv_export:
            try:
                with open(self.csv_filename, 'w', newline='') as csvfile:
                    fieldnames = ['frame']

                    # 11번부터 32번 키포인트까지의 x, y, z 컬럼 이름 생성
                    for i in range(11, 33):
                        fieldnames.extend([f'kp{i}_x', f'kp{i}_y', f'kp{i}_z'])

                    # CoM 좌표
                    fieldnames.extend(['com_x', 'com_y', 'com_z'])

                    # 예측 클래스 및 확률
                    fieldnames.extend(['predicted_class', 'predicted_class_name'])
                    if self.binary_mode:
                        fieldnames.extend(['prob_standing', 'prob_walking'])
                    else:
                        fieldnames.extend(['prob_standing', 'prob_start_walking', 'prob_walking', 'prob_finish_walking'])

                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                print(f"CSV 파일 {self.csv_filename}이 생성되었습니다.")
            except Exception as e:
                print(f"CSV 파일 초기화 중 오류 발생: {e}")
                self.csv_export = False

    def process_keypoints(self, keypoints_3d, com_3d, frame_id=None):
        """
        키포인트와 CoM을 처리하여 행동 예측

        Args:
            keypoints_3d: 3D 키포인트
            com_3d: 3D CoM 좌표
            frame_id: 현재 프레임 ID

        Returns:
            tuple: (예측 클래스, 클래스 이름, 확률 분포, 예측 상태)
        """
        if frame_id is None:
            self.frame_counter += 1
            frame_id = self.frame_counter

        prediction_status = "normal"  # 예측 상태: normal, collecting, error

        try:
            # 측정 시작
            prediction_start_time = time.time()

            # 유효한 키포인트 및 CoM 검사
            if keypoints_3d is None or com_3d is None:
                return None, "No valid keypoints", [0.25, 0.25, 0.25, 0.25], "error"

            # 인덱스 유효성 검사 추가
            if len(keypoints_3d) >= 33:
                # 필터링된 키포인트 중에서 11번부터 32번까지 추출
                keypoints_for_prediction = keypoints_3d[11:33]  # 22개 키포인트
            else:
                # 키포인트 수가 충분하지 않은 경우
                print(f"경고: 키포인트 수 부족 ({len(keypoints_3d)}개, 33개 필요)")
                if len(keypoints_3d) > 11:
                    keypoints_for_prediction = keypoints_3d[11:]
                else:
                    keypoints_for_prediction = keypoints_3d.copy()

            # CoM 3D 좌표 추가
            keypoints_and_com = list(keypoints_for_prediction)
            keypoints_and_com.append(com_3d)

            # 키포인트 전처리 - 정규화 적용
            processed_keypoints = self.tft_keypoint_processor.process(keypoints_and_com)

            # CSV 내보내기를 위한 정규화된 키포인트 데이터 준비
            if self.csv_export:
                # 내보내기를 위한 데이터 딕셔너리 초기화
                csv_data = {'frame': frame_id}

                # 정규화된 키포인트 데이터 저장 (11번부터 32번까지)
                valid_kp_count = len(processed_keypoints) - 1  # CoM을 제외한 키포인트 수
                for i in range(valid_kp_count):
                    kp_idx = i + 11  # 11번 키포인트부터 시작
                    if i < valid_kp_count:
                        kp = processed_keypoints[i]
                        csv_data[f'kp{kp_idx}_x'] = kp[0]
                        csv_data[f'kp{kp_idx}_y'] = kp[1]
                        csv_data[f'kp{kp_idx}_z'] = kp[2]
                    else:
                        # 키포인트가 없는 경우 0으로 채움
                        csv_data[f'kp{kp_idx}_x'] = 0
                        csv_data[f'kp{kp_idx}_y'] = 0
                        csv_data[f'kp{kp_idx}_z'] = 0

                # CoM 데이터는 정규화된 리스트의 마지막 항목
                if len(processed_keypoints) > 0:
                    com_processed = processed_keypoints[-1]
                    csv_data['com_x'] = com_processed[0]
                    csv_data['com_y'] = com_processed[1]
                    csv_data['com_z'] = com_processed[2]
                else:
                    csv_data['com_x'] = 0
                    csv_data['com_y'] = 0
                    csv_data['com_z'] = 0

            # 프레임 데이터 추가
            df = self.behavior_predictor.prepare_data(processed_keypoints, frame_id)

            # 프레임 버퍼에 충분한 데이터가 쌓였는지 확인
            if len(self.behavior_predictor.frame_buffer) < self.behavior_predictor.max_encoder_length:
                prediction_status = "collecting"
                predicted_class_name = f"Collecting data... ({len(self.behavior_predictor.frame_buffer)}/{self.behavior_predictor.max_encoder_length})"
                # 바이너리 모드에 따라 확률 조정
                if hasattr(self.behavior_predictor, 'binary_mode') and self.behavior_predictor.binary_mode:
                    probabilities = [0.5, 0.5]
                else:
                    probabilities = [0.25, 0.25, 0.25, 0.25]
                predicted_class = None

            else:
                # 충분한 데이터가 쌓인 경우에만 predict 호출
                predicted_class, predicted_class_name, probabilities = self.behavior_predictor.predict()

                # 예측 결과가 None이면 아직 데이터 수집 중
                if predicted_class is None:
                    prediction_status = "collecting"
                else:
                    # 예측 버퍼에 추가 (스무딩을 위해)
                    self.prediction_buffer.append(predicted_class)

                    # 스무딩된 예측 (가장 빈번한 클래스)
                    if len(self.prediction_buffer) > 0:
                        num_classes = len(probabilities)  # 확률 배열 길이로 클래스 수 결정
                        class_counts = np.bincount(list(self.prediction_buffer), minlength=num_classes)
                        smoothed_class = np.argmax(class_counts)

                        # 단순 다수결보다 더 안정적인 방식으로 스무딩 적용
                        # 최근 프레임의 예측에 더 높은 가중치 부여
                        weighted_counts = np.zeros(num_classes)
                        buffer_list = list(self.prediction_buffer)
                        for i, pred_class in enumerate(buffer_list):
                            # 최근 예측에 더 높은 가중치 부여 (선형 증가)
                            weight = (i + 1) / len(buffer_list)
                            weighted_counts[pred_class] += weight

                        smoothed_class = np.argmax(weighted_counts)
                        predicted_class = smoothed_class
                        predicted_class_name = self.behavior_predictor.class_names[smoothed_class]

                        # 확률 조정 (원본 확률과 스무딩된 클래스 간 균형)
                        if np.argmax(probabilities) != smoothed_class:
                            # 원래 확률 저장
                            original_probs = np.array(probabilities).copy()

                            # 스무딩된 클래스에 가중치 부여하여 확률 조정
                            alpha = 0.3  # 스무딩 정도
                            adjusted_probs = original_probs.copy()
                            max_idx = np.argmax(original_probs)

                            # 최대 확률값을 감소시키고 스무딩된 클래스의 확률값 증가
                            adjusted_probs[max_idx] -= alpha * original_probs[max_idx]
                            adjusted_probs[smoothed_class] += alpha * original_probs[max_idx]

                            # 확률의 합이 1이 되도록 정규화
                            adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                            probabilities = adjusted_probs.tolist()

            # CSV에 예측 결과 추가
            if self.csv_export:
                # 예측 클래스와 클래스 이름 저장
                csv_data['predicted_class'] = predicted_class if predicted_class is not None else -1
                csv_data['predicted_class_name'] = predicted_class_name

                # 확률 저장
                if hasattr(self.behavior_predictor, 'binary_mode') and self.behavior_predictor.binary_mode:
                    csv_data['prob_standing'] = probabilities[0]
                    csv_data['prob_walking'] = probabilities[1]
                else:
                    csv_data['prob_standing'] = probabilities[0]
                    csv_data['prob_start_walking'] = probabilities[1]
                    csv_data['prob_walking'] = probabilities[2]
                    csv_data['prob_finish_walking'] = probabilities[3]

                # CSV 파일에 데이터 추가
                try:
                    with open(self.csv_filename, 'a', newline='') as csvfile:
                        fieldnames = list(csv_data.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(csv_data)
                except Exception as e:
                    print(f"CSV 파일 쓰기 중 오류 발생: {e}")

            # 예측 완료 시간 측정
            prediction_end_time = time.time()
            prediction_time = prediction_end_time - prediction_start_time
            self.perf_monitor['prediction_times'].append(prediction_time)

            return predicted_class, predicted_class_name, probabilities, prediction_status

        except Exception as e:
            print(f"예측 처리 중 오류: {e}")
            traceback.print_exc()

            # 바이너리 모드에 따라 기본값 변경
            if hasattr(self.behavior_predictor, 'binary_mode') and self.behavior_predictor.binary_mode:
                return None, "Error: " + str(e)[:50], [0.5, 0.5], "error"
            else:
                return None, "Error: " + str(e)[:50], [0.25, 0.25, 0.25, 0.25], "error"

    def visualize_prediction(self, frame, predicted_class, predicted_class_name, probabilities, prediction_status):
        """
        행동 예측 결과를 화면에 시각화 - Walking은 파란색, Standing은 빨간색으로 표시

        Args:
            frame: 시각화할 프레임
            predicted_class: 예측 클래스 인덱스
            predicted_class_name: 예측 클래스 이름
            probabilities: 각 클래스별 확률
            prediction_status: 예측 상태 (normal, collecting, error)

        Returns:
            numpy.ndarray: 시각화된 프레임
        """
        # 행동 예측 결과 시각화
        if prediction_status == "normal":
            prediction_text = f"Behavior: {predicted_class_name}"

            # 바이너리 모드에서 예측된 클래스에 따라 색상 설정 (Walking: 파란색, Standing: 빨간색)
            if hasattr(self, 'binary_mode') and self.binary_mode:
                if predicted_class_name == "Walking":
                    text_color = (255, 0, 0)  # BGR 형식: 파란색
                else:  # Standing
                    text_color = (0, 0, 255)  # BGR 형식: 빨간색
            else:
                text_color = (0, 0, 255)  # 기본값: 빨간색

        elif prediction_status == "collecting":
            prediction_text = f"Behavior: {predicted_class_name}"
            text_color = (0, 165, 255)  # 주황색
        else:  # error
            prediction_text = f"Behavior: Error"
            text_color = (0, 0, 255)  # 빨간색

        cv2.putText(frame, prediction_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        # 확률 시각화 (바 차트 형태로)
        if sum(probabilities) > 0 and predicted_class is not None:
            bar_height = 20
            bar_max_width = 150
            bar_x = 20
            bar_y = 70

            # 바이너리 모드이면 클래스 이름을 'Standing'과 'Walking'으로 설정
            class_names = ['Standing', 'Walking'] if hasattr(self, 'binary_mode') and self.binary_mode \
                        else self.behavior_predictor.class_names

            for i, (prob, class_name) in enumerate(zip(probabilities, class_names)):
                bar_width = int(prob * bar_max_width)

                # 클래스에 따라 색상 설정
                if hasattr(self, 'binary_mode') and self.binary_mode:
                    if class_name == "Walking":
                        color = (255, 0, 0) if i == predicted_class else (128, 0, 0)  # 파란색/어두운 파란색
                    else:  # Standing
                        color = (0, 0, 255) if i == predicted_class else (0, 0, 128)  # 빨간색/어두운 빨간색
                else:
                    color = (0, 255, 0) if i == predicted_class else (0, 165, 255)  # 기본 색상

                cv2.rectangle(frame,
                            (bar_x, bar_y + i * (bar_height + 5)),
                            (bar_x + bar_width, bar_y + i * (bar_height + 5) + bar_height),
                            color, -1)

                text = f"{class_name}: {prob:.2f}"

                # 클래스에 따라 텍스트 색상 설정
                if hasattr(self, 'binary_mode') and self.binary_mode:
                    if class_name == "Walking":
                        text_color = (255, 0, 0)  # 파란색
                    else:  # Standing
                        text_color = (0, 0, 255)  # 빨간색
                else:
                    text_color = (255, 255, 255)  # 흰색

                cv2.putText(frame, text,
                        (bar_x + bar_max_width + 10, bar_y + i * (bar_height + 5) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        return frame

    def reset(self):
        """버퍼 및 예측기 초기화"""
        self.behavior_predictor.reset()
        self.prediction_buffer.clear()
        self.frame_counter = 0
