import logging
import time

import cv2
import numpy as np
from aiologger import Logger
from aiologger.levels import LogLevel
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation

from model import DetectorState, PipelineSettings, UISettings, VisionSegment

async_log = Logger.with_default_handlers(
    name="photonvision_detector", level=LogLevel.INFO
)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("photonvision_detector")


def init_detector(p: PipelineSettings) -> Detector:
    return Detector(
        families=p.family,
        nthreads=p.nthreads,
        quad_decimate=p.quad_decimate,
        quad_sigma=p.quad_sigma,
        refine_edges=p.refine_edges,
        decode_sharpening=p.decode_sharpening,
    )


def process_frame(
    frame_bgr: np.ndarray,
    app_config: UISettings,
    detector_state: DetectorState,
    camera_state: VisionSegment,
    field_tag_poses,
    capture_timestamp: int,
    draw_ui_elements: bool = False,
):
    t0 = time.time()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    res_key = f"{gray.shape[1]}x{gray.shape[0]}"
    calib = app_config.calibration.size_calib_data.get(res_key)
    if not calib:
        K = np.array(
            [[800, 0, gray.shape[1] // 2], [0, 800, gray.shape[0] // 2], [0, 0, 1]],
        )
        dist = np.array([0] * 4)
    else:
        K = calib.cameraMatrix
        dist = calib.distCoeffs

    detections = detector_state.detector.detect(gray)

    obj_pts = []
    img_pts = []
    detected_targets: list[int] = []

    for d in detections:
        if d.tag_id not in field_tag_poses:
            continue
        detected_targets.append(d.tag_id)
        if draw_ui_elements:
            # Draw tag
            corners = d.corners.astype(int)
            for i in range(4):
                cv2.line(
                    display,
                    tuple(corners[i]),
                    tuple(corners[(i + 1) % 4]),
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                display,
                f"ID:{d.tag_id}",
                tuple(d.center.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # Local corners
        s = app_config.global_data.tag_size_m
        local = np.array(
            [
                [0, -s / 2, -s / 2],
                [0, s / 2, -s / 2],
                [0, s / 2, s / 2],
                [0, -s / 2, s / 2],
            ],
            dtype=np.float32,
        )

        pose = field_tag_poses[d.tag_id]
        t = np.array(
            [
                pose["translation"]["x"],
                pose["translation"]["y"],
                pose["translation"]["z"],
            ]
        )
        q = [
            pose["rotation"]["quaternion"]["W"],
            pose["rotation"]["quaternion"]["X"],
            pose["rotation"]["quaternion"]["Y"],
            pose["rotation"]["quaternion"]["Z"],
        ]
        R = Rotation.from_quat(q, scalar_first=True).as_matrix()

        for pt in local:
            obj_pts.append(R @ pt + t)
        img_pts.extend(d.corners)

        if draw_ui_elements:
            # Per-tag PnP for axes
            ok, r, t = cv2.solvePnP(
                local.reshape(-1, 1, 3), d.corners.reshape(-1, 1, 2), K, dist
            )
            if ok:
                axis = np.float32([[0, 0, 0], [0.06, 0, 0], [0, 0.06, 0], [0, 0, 0.06]])
                pts, _ = cv2.projectPoints(axis, r, t, K, dist)
                pts = pts.astype(int).reshape(-1, 2)
                o = tuple(pts[0])
                cv2.line(display, o, tuple(pts[1]), (0, 0, 255), 3)  # X
                cv2.line(display, o, tuple(pts[2]), (0, 255, 0), 3)  # Y
                cv2.line(display, o, tuple(pts[3]), (255, 0, 0), 3)  # Z

    # Multi-tag PnP
    T_field_robot = None
    if len(obj_pts) >= 4:
        obj = np.array(obj_pts, dtype=np.float32)
        img = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
        ok, r, t = cv2.solvePnP(obj, img, K, dist)
        if ok:
            R_cam, _ = cv2.Rodrigues(r)
            T_cam_field = np.eye(4)
            T_cam_field[:3, :3] = R_cam
            T_cam_field[:3, 3] = t.flatten()
            T_field_cam = np.linalg.inv(T_cam_field)

            # Camera → Robot
            yaw = np.deg2rad(app_config.robot_offset.yaw)
            R_cr = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1],
                ]
            )
            t_cr = np.array(
                [
                    app_config.robot_offset.tx,
                    app_config.robot_offset.ty,
                    app_config.robot_offset.tz,
                ]
            )
            T_cam_robot = np.eye(4)
            T_cam_robot[:3, :3] = R_cr
            T_cam_robot[:3, 3] = t_cr

            T_field_robot = T_field_cam @ T_cam_robot

    if draw_ui_elements:
        # Mini-map
        map_w, map_h = 256, 128
        scale = min(map_w / 16.5, map_h / 8.2)
        map_img = np.full((map_h, map_w, 3), 40, dtype=np.uint8)
        ox = (map_w - int(16.5 * scale)) // 2
        oy = (map_h - int(8.2 * scale)) // 2
        cv2.rectangle(
            map_img,
            (ox, oy),
            (ox + int(16.5 * scale), oy + int(8.2 * scale)),
            (255, 255, 255),
            1,
        )
        for tid, p in field_tag_poses.items():
            px = int(ox + p["translation"]["x"] * scale)
            py = int(oy + p["translation"]["y"] * scale)
            cv2.circle(map_img, (px, py), 2, (0, 255, 0), -1)
        if T_field_robot is not None:
            pos = T_field_robot[:3, 3]
            px = int(ox + pos[0] * scale)
            py = int(oy + pos[1] * scale)
            cv2.circle(map_img, (px, py), 4, (0, 0, 255), -1)
            yaw = np.arctan2(T_field_robot[1, 0], T_field_robot[0, 0])
            dx = 12 * np.cos(yaw)
            dy = 12 * np.sin(yaw)
            cv2.arrowedLine(
                map_img, (px, py), (int(px + dx), int(py + dy)), (255, 0, 0), 1
            )
        h, w = display.shape[:2]
        display[0:map_h, w - map_w : w] = cv2.resize(map_img, (map_w, map_h))

    # Stats
    latency = (time.time() - t0) * 1000
    if draw_ui_elements:
        cv2.putText(
            display,
            f"Latency: {latency:.1f}ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            display,
            f"Tag Size: {app_config.global_data.tag_size_m:.4f}m",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            display,
            f"Tags: {len(detections)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    camera_state.last_pose = T_field_robot
    camera_state.last_latency = latency
    camera_state.current_frame_processed = display
    camera_state.last_ids = detected_targets
    camera_state.last_capture_time = capture_timestamp

    return [d.tag_id for d in detections]
