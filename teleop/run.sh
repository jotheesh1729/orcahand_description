#!/bin/bash
# Run from repo root: bash teleop/run.sh
cd "$(dirname "$0")/.."

python teleop/camera_server.py &
CAMERA_PID=$!

mjpython teleop/mujoco_viewer.py

kill $CAMERA_PID 2>/dev/null
