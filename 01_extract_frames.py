import cv2
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------
VIDEO_BASE_DIR = Path(
    "/data/Deep_Angiography/DICOM2Video/Code/Data_Processing/videos_out"
)

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


# -----------------------------
# Core logic
# -----------------------------
def extract_frames_1fps(video_path: Path):
    """
    Extracts 1 frame per second from a video and saves frames
    to a directory named after the video (without extension).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[WARN] Invalid FPS for video: {video_path}")
        cap.release()
        return

    frame_interval = int(round(fps))  # ~1 frame per second

    out_dir = video_path.parent / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = out_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[OK] {video_path.name}: saved {saved_idx} frames → {out_dir}")


def main():
    if not VIDEO_BASE_DIR.exists():
        raise FileNotFoundError(f"Base path not found: {VIDEO_BASE_DIR}")

    videos = [
        p for p in VIDEO_BASE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    print(f"Found {len(videos)} videos in {VIDEO_BASE_DIR}")

    for video_path in videos:
        extract_frames_1fps(video_path)


if __name__ == "__main__":
    main()
