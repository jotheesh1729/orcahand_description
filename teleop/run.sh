#!/bin/bash
# Run from repo root: bash teleop/run.sh [camera_index]
#
# Examples:
#   bash teleop/run.sh        (auto-detects camera)
#   bash teleop/run.sh 0      (MacBook webcam)
#   bash teleop/run.sh 1      (RealSense D455)

cd "$(dirname "$0")/.."

python teleop/camera_server.py "$@" &
CAMERA_PID=$!

mjpython teleop/mujoco_viewer.py

kill $CAMERA_PID 2>/dev/null
