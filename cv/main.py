#!/usr/bin/env python3
import cv2
import numpy as np
import time
import os
import requests
import json
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------
# Quaternion to Rotation Matrix
# --------------------------------------------------------------
def quat_to_rotmat(q):
    """Convert quaternion [W, X, Y, Z] to 3x3 rotation matrix."""
    return np.array(Rotation.from_quat(q, scalar_first=True).as_matrix())


# --------------------------------------------------------------
# 1. Fetch FRC 2025 Field Layout
# --------------------------------------------------------------
url = "https://raw.githubusercontent.com/wpilibsuite/allwpilib/main/apriltag/src/main/native/resources/edu/wpi/first/apriltag/2025-reefscape-welded.json"
try:
    response = requests.get(url)
    response.raise_for_status()
    field_layout = response.json()
    tags = field_layout["tags"]
    tag_poses = {tag["ID"]: tag["pose"] for tag in tags}
    print(f"Loaded {len(tags)} AprilTags from FRC 2025 Reefscape field layout.")
except Exception as e:
    print(f"Failed to fetch field layout: {e}")
    tags = []
    tag_poses = {}

# --------------------------------------------------------------
# 2. Multi-core setup
# --------------------------------------------------------------
nthreads = 4
print(f"Using {nthreads} CPU threads for detection")

# --------------------------------------------------------------
# 3. FRC-specific parameters
# --------------------------------------------------------------
tagsize = 0.1524  # 6 inches in meters
family = "tag36h11"
field_length = 16.4592  # X
field_width = 8.2296  # Y

fx = fy = 800.0
cx = 640.0
cy = 480.0

# --------------------------------------------------------------
# 4. Camera to Robot Transform (T_cam_robot: robot origin in cam frame)
# --------------------------------------------------------------
cam_tx = 0.0  # right (+X in cam)
cam_ty = 0.0  # down (+Y in cam)
cam_tz = 0.0  # forward (+Z in cam)
cam_yaw_deg = 0.0

yaw_rad = np.deg2rad(cam_yaw_deg)
R_cam_robot = np.array(
    [
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1],
    ]
)
t_cam_robot = np.array([cam_tx, cam_ty, cam_tz])
T_cam_robot = np.vstack(
    (np.hstack((R_cam_robot, t_cam_robot[:, np.newaxis])), [0, 0, 0, 1])
)

print("Camera to robot transform applied.")

# --------------------------------------------------------------
# 5. Detector
# --------------------------------------------------------------
detector = Detector(
    families=family,
    nthreads=nthreads,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.8,
    debug=0,
)

# --------------------------------------------------------------
# 6. Open webcam
# --------------------------------------------------------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 120)

cv2.namedWindow("AprilTag – Grayscale (FRC Localization)", cv2.WINDOW_NORMAL)

# --------------------------------------------------------------
# 7. Mini-map in corner setup
# --------------------------------------------------------------
map_w, map_h = 256, 128  # Small for corner
scale = min(map_w / field_length, map_h / field_width)

# --------------------------------------------------------------
# 8. Timing
# --------------------------------------------------------------
prev_time = time.time()
fps = 0.0

print("FRC 2025 AprilTag Localization – press 'q' to quit")

while True:
    t0 = time.time()

    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray is None:
        continue

    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 3-channel for drawing

    detections = detector.detect(gray)

    # --------------------------------------------------------------
    # 9. Collect for multi-tag PnP (adapted from WPILib)
    # --------------------------------------------------------------
    obj_field = []
    img_corners = []
    known_tag_ids = []

    for det in detections:
        tag_id = det.tag_id
        if tag_id in tag_poses:
            # Draw tag overlay
            corners = det.corners.astype(int)
            for i in range(4):
                cv2.line(
                    display,
                    tuple(corners[i]),
                    tuple(corners[(i + 1) % 4]),
                    (0, 255, 0),
                    3,
                )
            centre = tuple(det.center.astype(int))
            cv2.putText(
                display,
                f"ID:{tag_id}",
                (centre[0] - 30, centre[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Local tag corners (bottom-left start, Y up: bl, br, tr, tl)
            local_corners = np.array(
                [
                    [0, -tagsize / 2, -tagsize / 2],
                    [0, tagsize / 2, -tagsize / 2],
                    [0, tagsize / 2, tagsize / 2],
                    [0, -tagsize / 2, tagsize / 2],
                ],
                dtype=np.float32,
            )

            # Field corners for this tag
            pose = tag_poses[tag_id]
            t_field_tag = np.array(
                [
                    pose["translation"]["x"],
                    pose["translation"]["y"],
                    pose["translation"]["z"],
                ]
            )
            q = np.array(
                [
                    pose["rotation"]["quaternion"]["W"],
                    pose["rotation"]["quaternion"]["X"],
                    pose["rotation"]["quaternion"]["Y"],
                    pose["rotation"]["quaternion"]["Z"],
                ]
            )
            # print(q)
            R_field_tag = quat_to_rotmat(q)
            total_mat = np.vstack(
                (np.hstack((R_field_tag, t_field_tag[:, np.newaxis])), [0, 0, 0, 1])
            )

            for local_pt in local_corners:
                # print(total_mat, np.hstack((local_pt, [1,1,1,1])))
                field_pt = total_mat @ np.hstack((local_pt, [1]))
                # field_pt = R_field_tag @ local_pt + t_field_tag
                # + t_field_tag
                obj_field.append(field_pt[:3])
            #   print(field_pt)

            # Image corners (assume order matches: adjust if needed)
            for corner in det.corners:
                img_corners.append(corner)

            known_tag_ids.append(tag_id)

            # Draw axes (local tag pose)
            obj_pts_img = det.corners.reshape(-1, 1, 2).astype(np.float32)
            obj_pts_local = local_corners.reshape(-1, 1, 3).astype(np.float32)
            cam_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            dist = np.zeros((4, 1))
            success, rvec, tvec = cv2.solvePnP(
                obj_pts_local, obj_pts_img, cam_mat, dist
            )
            if success:
                axis_len = 0.06
                axis3d = np.float32(
                    [[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]
                )
                imgpts, _ = cv2.projectPoints(axis3d, rvec, tvec, cam_mat, dist)
                imgpts = imgpts.astype(int).reshape(-1, 2)
                origin = tuple(imgpts[0].ravel())
                cv2.line(
                    display, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 4
                )  # X red
                cv2.line(
                    display, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 4
                )  # Y green
                cv2.line(
                    display, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 4
                )  # Z blue

    num_tags = len(known_tag_ids)
    robot_pos = None
    robot_yaw = None

    if num_tags > 0 and len(obj_field) == 4 * num_tags:
        obj_field = np.array(obj_field, dtype=np.float32)
        img_corners = np.array(img_corners, dtype=np.float32).reshape(-1, 1, 2)

        success, rvec, tvec = cv2.solvePnP(obj_field, img_corners, cam_mat, dist)
        if success:
            R_cam_field, _ = cv2.Rodrigues(rvec)
            t_cam_field = tvec.flatten()
            T_cam_field = np.vstack(
                (np.hstack((R_cam_field, t_cam_field[:, np.newaxis])), [0, 0, 0, 1])
            )
            T_field_cam = np.linalg.inv(T_cam_field)

            # Apply camera-to-robot
            T_field_robot = T_field_cam @ T_cam_robot

            # Robot pose in field
            robot_pos = T_field_robot[:3, 3]
            R_rf = T_field_robot[:3, :3]
            robot_yaw = np.arctan2(R_rf[1, 0], R_rf[0, 0])

            # Backside/height check and flip if needed
            if robot_pos[2] < 0.1:  # Assuming camera above field
                print(f"Flipping pose (low Z: {robot_pos[2]:.2f})")
                flip_R = np.array(
                    [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
                )  # 180 deg around X
                T_field_robot[:3, :3] = R_rf @ flip_R
                T_field_robot[:3, 3] = -robot_pos  # Flip position
                robot_pos = T_field_robot[:3, 3]
                R_rf = T_field_robot[:3, :3]
                robot_yaw = np.arctan2(R_rf[1, 0], R_rf[0, 0])

    # --------------------------------------------------------------
    # 11. Stats
    # --------------------------------------------------------------
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000.0

    now = time.time()
    fps = 0.9 * fps + 0.1 / (t1 - t0)
    prev_time = now

    # --------------------------------------------------------------
    # 10. Draw mini field in top-right corner
    # --------------------------------------------------------------
    map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8) * 50

    # Field outline (centered)
    field_px_w = int(field_length * scale)
    field_px_h = int(field_width * scale)
    offset_x = (map_w - field_px_w) // 2
    offset_y = (map_h - field_px_h) // 2
    cv2.rectangle(
        map_img,
        (offset_x, offset_y),
        (offset_x + field_px_w, offset_y + field_px_h),
        (255, 255, 255),
        1,
    )

    # Tags
    for tag in tags:
        pose = tag["pose"]
        px = int(offset_x + pose["translation"]["x"] * scale)
        py = int(offset_y + pose["translation"]["y"] * scale)
        cv2.circle(map_img, (px, py), 2, (0, 255, 0), -1)
        cv2.putText(
            map_img,
            str(tag["ID"]),
            (px + 3, py + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )

        # Robot
    if robot_pos is not None:
        px = int(offset_x + robot_pos[0] * scale)
        py = int(offset_y + robot_pos[1] * scale)
        # print(px, py)
        cv2.circle(map_img, (px, py), 4, (0, 0, 255), -1)

        # Heading arrow
        arrow_len = 10
        dx = arrow_len * np.cos(robot_yaw)
        dy = arrow_len * np.sin(robot_yaw)
        cv2.arrowedLine(map_img, (px, py), (int(px + dx), int(py + dy)), (255, 0, 0), 1)
        print(
            f"Robot Pose (from {num_tags} tags): [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}] m, Yaw {np.rad2deg(robot_yaw):.1f}°"
        )
    # Overlay small map in top-right of display
    small_map = cv2.resize(map_img, (map_w, map_h))
    h, w = display.shape[:2]
    display[0:map_h, w - map_w : w] = small_map

    cv2.putText(
        display,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        display,
        f"Latency: {latency_ms:.1f} ms",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
    )
    cv2.putText(
        display,
        f"Tags: {num_tags}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.imshow("AprilTag – Grayscale (FRC Localization)", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
