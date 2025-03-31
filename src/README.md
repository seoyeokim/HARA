### Conda 가상환경 설정
- 실행 환경 : MAC OS (실리콘 M4 chip)
- local 에서 사용 시 각 환경에 맞는 conda 구성필요

❯ conda create --name kf_proj python=3.10 -y

❯ pip install torch torchvision torchaudio

❯ pip install tensorflow-macos tensorflow-metal

❯ pip install opencv-python mediapipe numpy

❯ pip install pandas

### TFT 모델 Installation
❯ pip install pytorch-forecasting

---

### Media Pipe Pose 실행 방법
- 3D pose
  - 카메라 사용: python main_3d.py 또는 python main_3d.py -i 0
  - 비디오 파일 처리: python main_3d.py -i path/to/video.mp4
  - ROI 비율 조정: python main_3d.py -i path/to/video.mp4 -r 0.7

### TFT 보행자 행동 추정
- 카메라 사용: python main_TFT_inference.py -i 0 or python main_TFT_inference.py -i 0 -o 출력파일.mp4
- 비디오 파일 처리: python main_TFT_inference.py -i 비디오파일.mp4
