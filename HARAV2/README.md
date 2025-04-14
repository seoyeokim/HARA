### Conda 가상환경 설정

❯ conda create --name kf_proj python=3.10 -y

❯ pip install -r requirements.txt

---

### Inference 실행 방법
- 카메라 사용: python main.py 또는 python main.py -i 0
- 비디오 파일 처리: python main.py -i path/to/video.mp4
- 모델기반 동적 Roi : python main.py -t -i 또는 python main.py -t -i path/to/video.mp4 (동적 Roi 재타게팅 시, Inference 영상에서 키보드 k 버튼 )
- 2 class 분류 : python main.py -b -i 또는 python main.py -b -i path/to/video.mp4 (-b 옵션을 지정하지 않을 경우, 4 class 분류)
- 커스텀 체크포인트 : python main.py --ckp-path path/to/checkpoint.pt -i 또는 python main.py --ckp-path path/to/checkpoint.pt -i path/to/video.mp4
