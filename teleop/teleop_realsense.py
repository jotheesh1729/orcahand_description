#!/usr/bin/env python3
"""
RealSense D455 + MediaPipe Hands → Orca Hand MuJoCo teleoperation

Usage:
    cd orcahand_description
    python3 teleop/teleop_realsense.py

Controls:
    q  — quit
"""

import threading
import numpy as np
import cv2
import mujoco
import mujoco.viewer
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
import pyrealsense2 as rs

SCENE_XML = "v1/scene_right.xml"

LM = mp_hands.HandLandmark


def _angle_at(a, b, c):
    """Angle in degrees at vertex b, between rays b→a and b→c."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _flex(deg):
    """Convert 3-point angle (180° = straight, 90° = fully curled) to flexion radians."""
    return np.radians(180.0 - deg)


def landmarks_to_ctrl(lms, model):
    """
    Map 21 MediaPipe hand landmarks to Orca hand actuator targets (radians).
    Returns a numpy array of length model.nu.
    """
    ctrl = np.zeros(model.nu)

    def set_act(name, value):
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx >= 0:
            lo, hi = model.actuator_ctrlrange[idx]
            ctrl[idx] = float(np.clip(value, lo, hi))

    # ── Index ─────────────────────────────────────────────────────────────────
    set_act("right_index_mcp_actuator",
            _flex(_angle_at(lms[LM.WRIST],             lms[LM.INDEX_FINGER_MCP], lms[LM.INDEX_FINGER_PIP])))
    set_act("right_index_pip_actuator",
            _flex(_angle_at(lms[LM.INDEX_FINGER_MCP],  lms[LM.INDEX_FINGER_PIP], lms[LM.INDEX_FINGER_DIP])))

    # ── Middle ────────────────────────────────────────────────────────────────
    set_act("right_middle_mcp_actuator",
            _flex(_angle_at(lms[LM.WRIST],              lms[LM.MIDDLE_FINGER_MCP], lms[LM.MIDDLE_FINGER_PIP])))
    set_act("right_middle_pip_actuator",
            _flex(_angle_at(lms[LM.MIDDLE_FINGER_MCP],  lms[LM.MIDDLE_FINGER_PIP], lms[LM.MIDDLE_FINGER_DIP])))

    # ── Ring ──────────────────────────────────────────────────────────────────
    set_act("right_ring_mcp_actuator",
            _flex(_angle_at(lms[LM.WRIST],           lms[LM.RING_FINGER_MCP], lms[LM.RING_FINGER_PIP])))
    set_act("right_ring_pip_actuator",
            _flex(_angle_at(lms[LM.RING_FINGER_MCP], lms[LM.RING_FINGER_PIP], lms[LM.RING_FINGER_DIP])))

    # ── Pinky ─────────────────────────────────────────────────────────────────
    set_act("right_pinky_mcp_actuator",
            _flex(_angle_at(lms[LM.WRIST],      lms[LM.PINKY_MCP], lms[LM.PINKY_PIP])))
    set_act("right_pinky_pip_actuator",
            _flex(_angle_at(lms[LM.PINKY_MCP],  lms[LM.PINKY_PIP], lms[LM.PINKY_DIP])))

    # ── Thumb ─────────────────────────────────────────────────────────────────
    set_act("right_thumb_pip_actuator",
            _flex(_angle_at(lms[LM.THUMB_CMC], lms[LM.THUMB_MCP], lms[LM.THUMB_IP])))
    set_act("right_thumb_dip_actuator",
            _flex(_angle_at(lms[LM.THUMB_MCP], lms[LM.THUMB_IP],  lms[LM.THUMB_TIP])))

    return ctrl


# ── Shared state between camera thread and MuJoCo thread ─────────────────────
_latest_ctrl = None
_ctrl_lock = threading.Lock()


def camera_thread(model):
    global _latest_ctrl

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    print("[camera] RealSense started. Show your right hand. Press q to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if result.multi_hand_landmarks:
                lms = result.multi_hand_landmarks[0].landmark
                ctrl = landmarks_to_ctrl(lms, model)
                with _ctrl_lock:
                    _latest_ctrl = ctrl
                mp_draw.draw_landmarks(img, result.multi_hand_landmarks[0],
                                       mp_hands.HAND_CONNECTIONS)

            cv2.imshow("RealSense — Hand Tracking (q to quit)", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


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
