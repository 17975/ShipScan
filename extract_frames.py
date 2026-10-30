"""
extract_frames.py - turn a video into training images.

Usage:
    python extract_frames.py my_video.mp4 apple
    python extract_frames.py my_video.mp4 apple 0.5     (one frame every 0.5 sec)

Frames are saved into dataset/<class_name>/.
"""
import os
import sys
import cv2

if len(sys.argv) < 3:
    raise SystemExit("Usage: python extract_frames.py <video_file> <class_name> [seconds_between_frames]")

video_path, class_name = sys.argv[1], sys.argv[2]
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

out_dir = os.path.join("dataset", class_name)
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
step = max(1, int(fps * interval))
existing = len([f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

frame_idx, saved = 0, 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % step == 0:
        saved += 1
        cv2.imwrite(os.path.join(out_dir, f"{class_name}_vid_{existing + saved:04d}.jpg"), frame)
    frame_idx += 1

cap.release()
print(f"Saved {saved} frames to {out_dir}/")
