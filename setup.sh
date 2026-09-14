#!/usr/bin/env bash
# Install dependencies and fix the opencv-python/opencv-python-headless
# conflict caused by deepface's transitive dependency on plain
# opencv-python. requirements.txt alone can't prevent this — pip installs
# whatever deepface declares regardless of what we list ourselves — so
# this script removes the conflicting package after install every time.
#
# Symptom if this isn't run: AttributeError: module 'cv2' has no
# attribute 'CascadeClassifier' (or similar) at OSINT pipeline startup.
set -euo pipefail

# Some dependencies (tensorflow, via deepface) are large (500MB+) wheels;
# default pip timeouts are too aggressive for slower/unstable networks.
export PIP_DEFAULT_TIMEOUT=120

echo "Installing requirements (this can take a while — tensorflow alone is ~500MB)..."
pip install -r requirements.txt --break-system-packages --retries 10

echo "Removing conflicting opencv-python (opencv-python-headless is what we actually want)..."
pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless --break-system-packages 2>/dev/null || true

echo "Force-reinstalling opencv-python-headless to restore any files the uninstall above deleted,"
echo "and to guarantee the pinned <5.0.0 version (5.x removed cv2.CascadeClassifier)..."
# No --no-deps: needs to reinstall numpy and any other real dependency
# opencv-python-headless requires, not skip them.
pip install --force-reinstall --retries 10 "opencv-python-headless>=4.10.0,<5.0.0" --break-system-packages

echo "Verifying opencv-python-headless is intact..."
python3 -c "
import cv2
assert hasattr(cv2, 'CascadeClassifier'), 'cv2.CascadeClassifier missing — opencv install is still broken'
assert hasattr(cv2, 'VideoCapture'), 'cv2.VideoCapture missing — opencv install is still broken'
print('OK — opencv-python-headless is working correctly:', cv2.__version__)
"

echo "Setup complete."
