import cv2
import argparse

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return total_frames, fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count total frames in a video file.")
    parser.add_argument("video_path", type=str, help="Path to the MP4 video file")
    args = parser.parse_args()

    total_frames, fps = get_total_frames(args.video_path)
    print(f"총 프레임 수 : {total_frames}")
    print(f"영상 FPS : {fps}")