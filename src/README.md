### Conda 가상환경 설정
- 실험 환경 : MAC OS (실리콘 M4 chip)
- local 에서 사용 시 각 환경에 맞는 conda 구성필요

❯ conda create --name kf_proj python=3.10 -y  
❯ pip install torch torchvision torchaudio  
❯ pip install tensorflow-macos tensorflow-metal   
❯ pip install opencv-python mediapipe numpy  

---

### 실행 방법
- 3D pose  
❯ python main_3d.py

- 2D pose  
❯ python main.py
