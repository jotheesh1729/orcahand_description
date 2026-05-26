#!/usr/bin/env python3
"""
MuJoCo viewer — run with mjpython on macOS
Receives joint angles from camera_server via UDP and drives the Orca hand.

    mjpython teleop/mujoco_viewer.py
"""

import socket
import numpy as np
import mujoco
import mujoco.viewer

SCENE_XML = "v2/scene_right.xml"
UDP_HOST  = "127.0.0.1"
UDP_PORT  = 5005


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)

    ctrl_size = model.nu * 8   # float64 = 8 bytes per element

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.setblocking(False)

    print(f"[mujoco] Listening on {UDP_HOST}:{UDP_PORT} — start camera_server.py")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            try:
                pkt, _ = sock.recvfrom(ctrl_size)
                ctrl = np.frombuffer(pkt, dtype=np.float64)
                if ctrl.shape[0] == model.nu:
                    data.ctrl[:] = ctrl
            except BlockingIOError:
                pass

            mujoco.mj_step(model, data)
            viewer.sync()

    sock.close()


if __name__ == "__main__":
    main()
