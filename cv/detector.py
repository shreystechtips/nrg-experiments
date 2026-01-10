import logging
import time

import cv2
import numpy as np
from aiologger import Logger
from aiologger.levels import LogLevel
from photonlibpy.estimation.targetModel import TargetModel
from photonlibpy.estimation.visionEstimation import VisionEstimation
from photonlibpy.targeting import PhotonTrackedTarget
from photonlibpy.targeting.multiTargetPNPResult import PnpResult
from photonlibpy.targeting.TargetCorner import TargetCorner
from pupil_apriltags import Detector
from robotpy_apriltag import AprilTagFieldLayout

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
    field_layout: AprilTagFieldLayout,
    capture_timestamp: int,
    draw_ui_elements: bool = False,
):
    t0 = time.time()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    res_key = f"{gray.shape[1]}x{gray.shape[0]}"
    calib = app_config.calibration.get(app_config.selected_camera, {}).get(res_key)
    if not calib:
        K = np.array(
            [[1050, 0, gray.shape[1] // 2], [0, 1050, gray.shape[0] // 2], [0, 0, 1]],
        )
        dist = np.array([0] * 4)
    else:
        K = np.array(calib.cameraMatrix)
        dist = np.array(calib.distCoeffs)

    detections = detector_state.detector.detect(gray)

    visTags: list[PhotonTrackedTarget] = list()
    available_tags = field_layout.getTags()
    available_ids = [tag.ID for tag in available_tags]
    for d in detections:
        if d.tag_id not in available_ids:
            continue
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
                [0, -s / 2, s / 2],
                [0, s / 2, s / 2],
                [0, s / 2, -s / 2],
            ],
            dtype=np.float32,
        )

        visTags.append(
            PhotonTrackedTarget(
                detectedCorners=[TargetCorner(*r) for r in d.corners],
                fiducialId=d.tag_id,
            )
        )

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
    pnp_result: PnpResult | None = None
    if len(visTags) > 0:
        pnp_result = VisionEstimation.estimateCamPosePNP(
            K, np.array(dist), visTags, field_layout, TargetModel.AprilTag36h11()
        )
        if pnp_result is None:
            log.info("failed to get transforms")

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
        for tag in available_tags:
            pose = tag.pose
            px = int(ox + pose.X() * scale)
            py = int(oy + pose.Y() * scale)
            cv2.circle(map_img, (px, py), 2, (0, 255, 0), -1)
        if pnp_result is not None:
            T_field_robot: np.ndarray = pnp_result.best.toMatrix()
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
    if True:
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
            f"Tags: {len(visTags)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    camera_state.last_pnp = pnp_result
    camera_state.last_latency = latency
    camera_state.current_frame_processed = display
    camera_state.last_targets = visTags
    camera_state.last_capture_time = capture_timestamp

    return [d.fiducialId for d in visTags]
