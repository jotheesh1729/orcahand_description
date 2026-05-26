<div style="display: flex; gap: 10px;">
<p align="center">
  <img src=".docs/orcadescription_header.png" alt="ORCA Hand Description Header" style="max-width: 100%; height: auto;" />
</p>

</div>

# Orcahand Description

This repository contains the description files for the Orcahand model (both URDF and MJCF).

Versioned model layouts live under:
- `v1/`: legacy hand description files, including the `extended` variants
- `v2/`: updated hand description files from the newer Fusion360 export

The `extended` version contains additional bodies (incl. inertial properties) such as the camera mount, the U2D2 board and fans.

## Example Usage
1. Clone the repository:
   ```bash
   git clone git@github.com:orcahand/orcahand_description.git
   cd orcahand_description
   ```
2. Install the required visualization dependencies:
   ```bash
   pip install mujoco
   ```
3. Simulate any version using MuJoCo:
   ```bash
   python -m mujoco.viewer --mjcf=$(pwd)/v1/scene_combined.xml
   python -m mujoco.viewer --mjcf=$(pwd)/v2/scene_combined.xml
   ```

## Note on Meshes
Visual meshes contain the following amount of faces:
- Main tower, camera & fans: 15'000
- Rest of base: 2'000
- Skin: 5'000
- Rest: 500

Collision meshes contain the following amount of faces:
- Main tower, camera & fans: 7500
- Rest of base: 1000
- Skin: 500
- Rest: 250

You can further reduce or mirror meshes using the `utils/mesh_utils.py` script. With the same script, you can also print their number of faces. Some extra dependencies are required:
```bash
cd utils
pip install -r utils_requirements.txt
```

We can also recommend the VSCode extension `mtsmfm.vscode-stl-viewer` for quickly visualizing STL meshes.

---

## Hand Teleoperation (Camera to MuJoCo)

The `teleop/` folder contains a real-time hand teleoperation system. You show your hand to a camera and the Orca hand in MuJoCo mirrors your pose.

### How It Works

The pipeline has three stages:

**Stage 1: Hand Detection**

A camera captures video frames. Each frame is passed to MediaPipe Hand Landmarker, which detects 21 landmarks (3D points) on your hand. These landmarks cover the wrist, knuckle, and tip of every finger.

**Stage 2: Joint Angle Calculation**

From the 21 landmarks we compute the angle at each joint. The method depends on the joint type.

For finger flexion (curl), we use a three-point angle. Given three consecutive landmarks A, B, C (for example wrist, knuckle, middle of finger), the angle at B is:

```
angle = arccos( dot(B-A, B-C) / (|B-A| * |B-C|) )
```

When the finger is straight the angle is 180 degrees. When fully curled it is around 90 degrees. We convert this to a flexion value in radians:

```
flexion = radians(180 - angle)
```

This gives 0 when the finger is straight and increases as the finger curls.

For the thumb we need two additional joints that the other fingers do not have.

**Thumb CMC (opposition):** The CMC joint rotates the thumb so it can face across the palm and meet the index finger. To compute this we first define the palm frame. We take the cross product of two vectors rooted at the wrist:

```
v1 = index_mcp - wrist
v2 = pinky_mcp - wrist
palm_normal = normalize(cross(v2, v1))
```

The palm normal is a unit vector pointing out of the back of the hand. We then measure how far the thumb metacarpal direction dips below this plane:

```
thumb_dir = normalize(thumb_mcp - thumb_cmc)
cmc_angle = arcsin(-dot(thumb_dir, palm_normal))
```

When the thumb lies flat in the same plane as the palm the result is near zero. When the thumb rotates to oppose the index finger the value increases.

**Thumb abduction:** Abduction is how far the thumb swings sideways away from or toward the index finger. We project both the thumb direction and the index finger direction onto the palm plane, then measure the angle between those projections:

```
thumb_proj = thumb_dir - dot(thumb_dir, palm_normal) * palm_normal
index_proj = index_dir - dot(index_dir, palm_normal) * palm_normal
abd_angle  = arccos(dot(normalize(thumb_proj), normalize(index_proj)))
```

**Stage 3: Drive MuJoCo**

All computed angles are clamped to the actuator control ranges defined in the v2 MJCF model and written to `data.ctrl`. MuJoCo's position-controlled actuators then drive each joint to the target angle.

### Architecture

Because MuJoCo on macOS requires `mjpython` (a special launcher) and OpenCV's window system cannot run inside `mjpython`, the system is split into two processes that communicate over a local UDP socket:

```
camera_server.py  (run with: python)
  reads camera, runs MediaPipe, computes joint angles, shows preview window
  sends joint angle array over UDP port 5005

mujoco_viewer.py  (run with: mjpython)
  receives joint angles from UDP
  writes to data.ctrl and steps the MuJoCo simulation
```

### Installation

```bash
pip install mujoco mediapipe opencv-python numpy
```

On Apple Silicon (M1/M2/M3) Mac, install pyrealsense2 via conda if you want RealSense support:

```bash
conda install -c conda-forge pyrealsense2
```

### Running

```bash
cd orcahand_description
bash teleop/run.sh
```

This opens two windows: the camera preview with the hand skeleton drawn, and the MuJoCo viewer. Press Q in the camera window to quit both.

### Intel RealSense D455 Note

On macOS, the RealSense D455 cannot be accessed via `pyrealsense2` due to a USB access restriction introduced in recent macOS versions (`RS2_USB_STATUS_ACCESS`). The system currently uses OpenCV `VideoCapture` to read the D455 color stream as a standard UVC camera (camera index 1 on a MacBook with the D455 connected).

On Linux, `pyrealsense2` works without restriction and you can replace the `cv2.VideoCapture` section in `camera_server.py` with a full RealSense pipeline to also access depth data.

### Joints Mapped

| Joint | Description |
|---|---|
| right_i-mcp, right_i-pip | Index finger curl |
| right_m-mcp, right_m-pip | Middle finger curl |
| right_r-mcp, right_r-pip | Ring finger curl |
| right_p-mcp, right_p-pip | Pinky finger curl |
| right_t-cmc | Thumb opposition (rotation across palm) |
| right_t-abd | Thumb abduction (sideways spread) |
| right_t-mcp, right_t-pip | Thumb curl |

### References

- MediaPipe Hand Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- GeoRT geometric hand retargeting (Facebook Research): https://github.com/facebookresearch/GeoRT
- Joint angle calculation from MediaPipe landmarks: https://github.com/TemugeB/joint_angles_calculate
- Thumb angle calculation discussion: https://github.com/google/mediapipe/issues/2999
- Biomimetic hand gesture learning in humanoid robot: https://pmc.ncbi.nlm.nih.gov/articles/PMC11295247/
- Vision-based hand shadowing for robotic manipulation: https://arxiv.org/pdf/2603.11383

---

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or support, please contact the maintainers of this repo or the Orcahand team via our website ([https://orcahand.com](https://orcahand.com)).
