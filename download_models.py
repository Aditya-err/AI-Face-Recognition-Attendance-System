"""Download the two ONNX face models used by the application.

    YuNet face detector   (~0.2 MB)  face_detection_yunet_2023mar.onnx
    SFace face recognizer (~37 MB)   face_recognition_sface_2021dec.onnx

The models come from the official OpenCV model zoo (opencv_zoo) and are
published under the Apache-2.0 license.  Files are saved into the ``models/``
folder; the app loads them from there.

Usage:
    python download_models.py
    python download_models.py --force    # re-download even if files exist
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

FILES = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-download models that already exist")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face models into: {MODELS_DIR}\n")

    for filename, url in FILES.items():
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > 1000 and not args.force:
            print(f"[skip] {filename} already exists")
            continue
        print(f"[get ] {filename}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(f"[FAIL] could not download {filename}: {exc}")
            print(f"       Download it manually from:\n       {url}\n"
                  f"       and save it as: {dest}")
            sys.exit(1)
        print(f"       saved {dest.stat().st_size:,} bytes")

    # Verify the files load with the installed OpenCV.
    try:
        import cv2
        detector = cv2.FaceDetectorYN.create(
            str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"),
            "", (320, 320))
        recognizer = cv2.FaceRecognizerSF.create(
            str(MODELS_DIR / "face_recognition_sface_2021dec.onnx"), "")
        detector.release() if hasattr(detector, "release") else None
        recognizer.release() if hasattr(recognizer, "release") else None
        print("\n[OK] Models load correctly with the installed OpenCV.")
    except Exception as exc:
        print(f"\n[WARN] Models were downloaded but OpenCV could not load "
              f"them:\n       {exc}\n"
              f"       Check that opencv-contrib-python is installed:\n"
              f"       pip install opencv-contrib-python")
        sys.exit(1)

    print("\nDone. Start the app with:  python run.py")


if __name__ == "__main__":
    main()
