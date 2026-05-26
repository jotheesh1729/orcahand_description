#!/usr/bin/env python3
"""
RealSense D455 + MediaPipe Hands → Orca Hand MuJoCo teleoperation

Usage:
    cd orcahand_description
    python teleop/teleop_realsense.py

Controls:
    q  — quit
"""

import os
import time
import threading
import urllib.request
import numpy as np
import cv2
import mujoco
import mujoco.viewer
import mediapipe as mp

CAMERA_INDEX = 1  # 0 = MacBook webcam, 1 = D455 color stream

SCENE_XML = "v2/scene_right.xml"
MODEL_PATH = "teleop/hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Landmark indices (0-20)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9,  10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[setup] Downloading hand landmark model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[setup] Done.")


def _angle_at(a, b, c):
    """Angle in degrees at vertex b, between rays b→a and b→c."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _flex(deg):
    """Straight finger = 180° → 0 rad. Curled finger = 90° → ~1.57 rad."""
    return np.radians(180.0 - deg)


def landmarks_to_ctrl(lms, model):
    ctrl = np.zeros(model.nu)

    def set_act(name, value):
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx >= 0:
            lo, hi = model.actuator_ctrlrange[idx]
            ctrl[idx] = float(np.clip(value, lo, hi))

    set_act("right_i-mcp_actuator", _flex(_angle_at(lms[WRIST],      lms[INDEX_MCP],  lms[INDEX_PIP])))
    set_act("right_i-pip_actuator", _flex(_angle_at(lms[INDEX_MCP],  lms[INDEX_PIP],  lms[INDEX_DIP])))

    set_act("right_m-mcp_actuator", _flex(_angle_at(lms[WRIST],      lms[MIDDLE_MCP], lms[MIDDLE_PIP])))
    set_act("right_m-pip_actuator", _flex(_angle_at(lms[MIDDLE_MCP], lms[MIDDLE_PIP], lms[MIDDLE_DIP])))

    set_act("right_r-mcp_actuator", _flex(_angle_at(lms[WRIST],      lms[RING_MCP],   lms[RING_PIP])))
    set_act("right_r-pip_actuator", _flex(_angle_at(lms[RING_MCP],   lms[RING_PIP],   lms[RING_DIP])))

    set_act("right_p-mcp_actuator", _flex(_angle_at(lms[WRIST],      lms[PINKY_MCP],  lms[PINKY_PIP])))
    set_act("right_p-pip_actuator", _flex(_angle_at(lms[PINKY_MCP],  lms[PINKY_PIP],  lms[PINKY_DIP])))

    set_act("right_t-mcp_actuator", _flex(_angle_at(lms[THUMB_CMC],  lms[THUMB_MCP],  lms[THUMB_IP])))
    set_act("right_t-pip_actuator", _flex(_angle_at(lms[THUMB_MCP],  lms[THUMB_IP],   lms[THUMB_TIP])))

    return ctrl


def _draw_landmarks(img, lms, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 200, 0), 2)
    for x, y in pts:
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)


_latest_ctrl = None
_ctrl_lock = threading.Lock()


def camera_thread(model):
    global _latest_ctrl

    _ensure_model()

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[camera] ERROR: could not open camera index {CAMERA_INDEX}")
        return

    print("[camera] D455 started. Show your right hand. Press q to quit.")
    with HandLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    continue

                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]
                    ctrl = landmarks_to_ctrl(lms, model)
                    with _ctrl_lock:
                        _latest_ctrl = ctrl
        finally:
            cap.release()


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    t = threading.Thread(target=camera_thread, args=(model,), daemon=True)
    t.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with _ctrl_lock:
                if _latest_ctrl is not None:
                    data.ctrl[:] = _latest_ctrl
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
