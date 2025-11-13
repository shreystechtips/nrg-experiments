#!/usr/bin/env python3
import asyncio
import base64
from os import RTLD_GLOBAL
import cv2
import json
import logging
import numpy as np
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic.networks import IPvAnyAddress

import websockets
from aiohttp import web, WSMsgType
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation
from networktables import NetworkTables
from cv2_enumerate_cameras import enumerate_cameras

# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("photonvision")

# --------------------------------------------------------------
# Persistence
# --------------------------------------------------------------
CONFIG_PATH = Path("photonvision_config.json")
_save_lock = threading.Lock()
_save_timer: Optional[threading.Timer] = None


class CameraSettings(BaseModel):
    exposure: int = 12000
    gain: int = 12
    brightness: int = 50
    white_balance: int = 4600
    auto_exposure: bool = False
    auto_white_balance: bool = True
    led_mode: int = 1


class PipelineSettings(BaseModel):
    family: str = "tag36h11"
    nthreads: int = 4
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: float = 1
    decode_sharpening: float = 0.8


class CalibrationData(BaseModel):
    cameraMatrix: list[float]
    distCoeffs: list[float]
    reprojectionError: float


class CalibrationSettings(BaseModel):
    camera_index: int = Field(default=1)
    size_calib_data: dict[str, CalibrationData] = Field(default={})


class RobotOffset(BaseModel):
    tx: float = Field(default=0)
    ty: float = Field(default=0)
    tz: float = Field(default=0)
    yaw: float = Field(default=0)


class GlobalSettings(BaseModel):
    tag_size_m: float = Field(default=0.1524)
    team_number: int = Field(default=948)
    nt_server_addr: str = Field(default="10.9.48.0")
    led_brightness: int = Field(default=0, ge=0, le=1)
    driver_mode: bool = Field(default=False)


class CalibConfig(BaseModel):
    board_type: str = Field(default="Charuco")
    tag_family: str = Field(default="DICT_4X4_1000")
    pattern_spacing_in: float = Field(default=1.0)
    marker_size_in: float = Field(default=0.75)
    board_width_sq: int = Field(default=8)
    board_height_sq: int = Field(default=8)


class UISettings(BaseModel):
    selected_camera: int = Field(default=0)
    camera_settings: CameraSettings = Field(default=CameraSettings())
    pipeline: PipelineSettings = Field(default=PipelineSettings())
    calibration: CalibrationSettings = Field(default=CalibrationSettings())
    robot_offset: RobotOffset = Field(default=RobotOffset())
    global_data: GlobalSettings = Field(default=GlobalSettings())
    calib_board: CalibConfig = Field(default=CalibConfig())


app_config = UISettings()


def _debounced_save():
    global _save_timer
    with _save_lock:
        if _save_timer:
            _save_timer.cancel()
        _save_timer = threading.Timer(0.5, _do_save)
        _save_timer.start()


def _do_save():
    tmp = CONFIG_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(app_config.json(), f, indent=2)
    tmp.replace(CONFIG_PATH)
    log.info("Config saved")
    # except Exception as e:
    #     log.error(f"Save failed: {e}")


def load_config():
    global app_config
    if not CONFIG_PATH.exists():
        log.info("No config – using defaults")
        return
    try:
        with CONFIG_PATH.open() as f:
            data = json.load(f)
            print(data)
            app_config = UISettings.model_validate_json(data)
        # init_calib_board()
        log.info("Config loaded")
    except Exception as e:
        log.error(f"Config load failed: {e}")


# --------------------------------------------------------------
# NetworkTables
# --------------------------------------------------------------
nt = NetworkTables
camera_table = nt.getTable("/photonvision/default")


def ntproperty(path: str, default):
    def getter():
        return camera_table.getNumber(path, default)

    def setter(v):
        camera_table.putNumber(path, v)
        _debounced_save()

    return property(getter, setter)


# --------------------------------------------------------------
# Global Settings (persisted)
# --------------------------------------------------------------
# camera_settings = {
#     "exposure": 12000,
#     "gain": 12,
#     "brightness": 50,
#     "white_balance": 4600,
#     "auto_exposure": False,
#     "auto_white_balance": True,
#     "led_mode": 1,
# }
# pipeline_settings = {
#     "family": "tag36h11",
#     "nthreads": 4,
#     "quad_decimate": 1.0,
#     "quad_sigma": 0.0,
#     "refine_edges": 1,
#     "decode_sharpening": 0.8,
# }

calib_captures: List[np.ndarray] = []
calib_board = None  # will be set in init_calib_board()


def init_calib_board():
    global calib_board
    cfg = app_config.calib_board
    if cfg.board_type == "Charuco":
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, cfg.tag_family)
        )
        board = cv2.aruco.CharucoBoard(
            (cfg.board_width_sq, cfg.board_height_sq),
            cfg.pattern_spacing_in * 0.0254,  # in → m
            cfg.marker_size_in * 0.0254,
            dictionary,
        )
    else:
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, cfg.tag_family)
        )
        board = cv2.aruco.GridBoard(
            (cfg.board_width_sq, cfg.board_height_sq),
            cfg.marker_size_in * 0.0254,
            cfg.pattern_spacing_in * 0.0254,
            dictionary,
        )
    # TODO Sort this out
    calib_board = board


def compute_calibration():
    global current_calibration
    if len(calib_captures) < 10:
        log.warning("Need at least 10 good frames")
        return False

    res = calib_captures[0].shape[1::-1]  # (w, h)
    res_key = f"{res[0]}x{res[1]}"

    all_charuco_corners = []
    all_charuco_ids = []
    all_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in calib_captures]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for gray in all_gray:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, calib_board.getDictionary())
        if len(corners) == 0:
            continue
        if isinstance(calib_board, cv2.aruco.CharucoBoard):
            n, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, calib_board
            )
            if charuco_ids is not None and len(charuco_ids) > 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
        else:
            obj_points, img_points = calib_board.getGridPoints(corners, ids)
            if len(img_points) > 4:
                all_charuco_corners.append(img_points)
                all_charuco_ids.append(np.arange(len(img_points)).reshape(-1, 1))

    if len(all_charuco_corners) < 5:
        log.warning("Not enough good frames")
        return False

    h, w = all_gray[0].shape
    K = np.array([[w, 0, w // 2], [0, w, h // 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        calib_board,
        (w, h),
        K,
        dist,
        criteria=criteria,
    )

    current_calibration[res_key] = {
        "cameraMatrix": K.tolist(),
        "distCoeffs": dist.tolist(),
        "reprojectionError": float(rms),
    }
    log.info(f"Calibration done: RMS={rms:.3f}")
    calib_captures.clear()
    _debounced_save()
    return True


current_calibration: Dict[str, Dict] = {}
calib_config = {
    "board_type": "Charuco",
    "tag_family": "DICT_4X4_1000",
    "pattern_spacing_in": 1.0,
    "marker_size_in": 0.75,
    "board_width_sq": 8,
    "board_height_sq": 8,
}

# --------------------------------------------------------------
# Camera Management
# --------------------------------------------------------------
available_cameras: List[Dict[str, Any]] = []
camera_lock = threading.Lock()
current_cap: Optional[cv2.VideoCapture] = None
current_frame_raw = None
current_frame_processed = None
last_pose = None
last_latency = 0.0


def discover_cameras():
    global available_cameras
    all_cams = enumerate_cameras()
    cams = []
    for cam in all_cams:
        cap = cv2.VideoCapture(cam.index)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cams.append({"index": cam.index, "name": cam.name, "res": f"{w}x{h}"})
            cap.release()
    available_cameras = cams
    log.info(f"Found {len(cams)} cameras")


def open_selected_camera():
    global current_cap
    with camera_lock:
        if current_cap:
            current_cap.release()
        cap = cv2.VideoCapture(app_config.selected_camera)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 60)
            current_cap = cap
            log.info(f"Opened camera {app_config.selected_camera}")
        else:
            log.error(f"Failed to open camera {app_config.selected_camera}")


def apply_camera_settings():
    if not current_cap:
        print("cannot apply")
        return
    s = app_config.camera_settings
    current_cap.set(cv2.CAP_PROP_EXPOSURE, s.exposure if not s.auto_exposure else -1)
    current_cap.set(cv2.CAP_PROP_GAIN, s.gain)
    current_cap.set(cv2.CAP_PROP_BRIGHTNESS, s.brightness)
    current_cap.set(
        cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
        s.white_balance if not s.auto_white_balance else -1,
    )


# --------------------------------------------------------------
# AprilTag Detector
# --------------------------------------------------------------
detector: Optional[Detector] = None


def init_detector():
    global detector
    p = app_config.pipeline
    detector = Detector(
        families=p.family,
        nthreads=p.nthreads,
        quad_decimate=p.quad_decimate,
        quad_sigma=p.quad_sigma,
        refine_edges=p.refine_edges,
        decode_sharpening=p.decode_sharpening,
    )


# --------------------------------------------------------------
# Field Layout
# --------------------------------------------------------------
FIELD_LAYOUT_URL = "https://raw.githubusercontent.com/wpilibsuite/allwpilib/main/apriltag/src/main/native/resources/edu/wpi/first/apriltag/2025-reefscape-welded.json"
field_tag_poses: Dict[int, Dict] = {}


def load_field_layout():
    global field_tag_poses
    try:
        import requests

        r = requests.get(FIELD_LAYOUT_URL, timeout=5)
        r.raise_for_status()
        data = r.json()
        field_tag_poses = {t["ID"]: t["pose"] for t in data.get("tags", [])}
        log.info(f"Loaded {len(field_tag_poses)} tags")
    except Exception as e:
        log.warning(f"Field layout failed: {e}")


# --------------------------------------------------------------
# Core Detection
# --------------------------------------------------------------
def process_frame(frame_bgr):
    global last_pose, last_latency, current_frame_processed
    t0 = time.time()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    res_key = f"{gray.shape[1]}x{gray.shape[0]}"
    calib = current_calibration.get(res_key, {})
    K = np.array(
        calib.get(
            "cameraMatrix",
            [[800, 0, gray.shape[1] // 2], [0, 800, gray.shape[0] // 2], [0, 0, 1]],
        )
    )
    dist = np.array(calib.get("distCoeffs", [0, 0, 0, 0]))

    detections = detector.detect(gray)

    obj_pts = []
    img_pts = []
    targets = []

    for d in detections:
        if d.tag_id not in field_tag_poses:
            continue
        # Draw tag
        corners = d.corners.astype(int)
        for i in range(4):
            cv2.line(
                display, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2
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
        cv2.arrowedLine(map_img, (px, py), (int(px + dx), int(py + dy)), (255, 0, 0), 1)
    h, w = display.shape[:2]
    display[0:map_h, w - map_w : w] = cv2.resize(map_img, (map_w, map_h))

    # Stats
    latency = (time.time() - t0) * 1000
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

    last_pose = T_field_robot
    last_latency = latency
    current_frame_processed = display

    return [d.tag_id for d in detections]


# --------------------------------------------------------------
# Video Capture Loop
# --------------------------------------------------------------
def video_loop():
    global current_frame_raw
    while True:
        if not current_cap or not current_cap.isOpened():
            time.sleep(0.1)
            continue
        ret, frame = current_cap.read()
        if not ret:
            continue
        current_frame_raw = frame.copy()
        if not app_config.global_data.driver_mode:
            process_frame(frame)
        time.sleep(0.001)


# --------------------------------------------------------------
# HTTP Server (MJPEG + static)
# --------------------------------------------------------------
async def mjpeg_handler(request):
    """
    Streams processed (or raw) video as multipart/x-mixed-replace.
    """
    boundary = "frame"
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": f"multipart/x-mixed-replace;boundary={boundary}",
            "Cache-Control": "no-cache",
            "Connection": "close",
            "Pragma": "no-cache",
        },
    )
    await response.prepare(request)

    while True:
        # Choose frame: processed if we have it, otherwise raw
        frame = (
            current_frame_processed
            if current_frame_processed is not None
            else current_frame_raw
        )
        if frame is None:
            await asyncio.sleep(0.1)
            continue

        # Encode to JPEG (NumPy → bytes)
        success, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not success:
            await asyncio.sleep(0.1)
            continue

        jpg_bytes = jpg.tobytes()  # <-- correct bytes
        header = (
            f"--{boundary}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpg_bytes)}\r\n\r\n"  # <-- correct length
        ).encode(
            "utf-8"
        )  # <-- pure bytes

        try:
            await response.write(header + jpg_bytes + b"\r\n")
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            break  # client went away

        await asyncio.sleep(1 / 30)  # ~30 FPS cap


# --------------------------------------------------------------
# WebSocket API
# --------------------------------------------------------------
ws_clients = set()


def make_config():
    print(app_config.model_dump())
    return app_config.model_dump() | {"type": "config"}


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws_clients.add(ws)
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            t = data.get("type")

            if t == "get_cameras":
                await ws.send_json({"type": "cameras", "list": available_cameras})

            elif t == "get_config":
                await ws.send_json(make_config())

            elif t == "select_camera":
                idx = data.get("index", 0)
                if any(c["index"] == idx for c in available_cameras):
                    global app_config
                    app_config.selected_camera = idx
                    open_selected_camera()
                    apply_camera_settings()
                    _debounced_save()
                    await broadcast({"type": "camera_selected", "index": idx})

            elif t == "config":
                app_config.camera_settings = app_config.camera_settings.model_copy(
                    update=data.get("camera", {})
                )
                apply_camera_settings()
                _debounced_save()

            elif t == "pipeline":
                app_config.pipeline = app_config.pipeline.model_copy(
                    update=data.get("settings", {})
                )
                init_detector()
                _debounced_save()

            elif t == "calib_start":
                # TODO: calibration routine
                pass

            elif t == "calib_config":
                calib_config.update(data.get("config", {}))
                init_calib_board()
                _debounced_save()

            elif t == "calib_capture":
                if current_frame_raw is not None:
                    calib_captures.append(current_frame_raw.model_copy())
                    await ws.send_json(
                        {"type": "calib_status", "count": len(calib_captures)}
                    )

            elif t == "calib_compute":
                success = compute_calibration()
                await ws.send_json({"type": "calib_result", "success": success})

            elif t == "calib_clear":
                calib_captures.clear()
                await ws.send_json({"type": "calib_status", "count": 0})

            elif t == "nt_server":
                # global nt_server_addr
                app_config.global_data.nt_server_addr = data.get("address", "")
                nt.shutdown()
                if app_config.global_data.nt_server_addr:
                    nt.initialize(server=app_config.global_data.nt_server_addr)
                else:
                    nt.initialize()
                _debounced_save()

            elif t == "global":
                # global tag_size_m, team_number, led_brightness, driver_mode
                g = data.get("global", {})
                if "tag_size_m" in g:
                    app_config.global_data.tag_size_m = g["tag_size_m"]
                if "team_number" in g:
                    app_config.global_data.team_number = g["team_number"]
                if "nt_server" in g:
                    app_config.global_data.nt_server_addr = g["nt_server"]
                if "led_brightness" in g:
                    app_config.global_data.led_brightness = g["led_brightness"]
                if "driver_mode" in g:
                    app_config.global_data.driver_mode = g["driver_mode"]
                _debounced_save()

    finally:
        ws_clients.discard(ws)
    return ws


async def broadcast_config():
    await broadcast(make_config())


async def broadcast(msg):
    if ws_clients:
        text = json.dumps(msg)
        await asyncio.gather(
            *(ws.send_str(text) for ws in ws_clients), return_exceptions=True
        )


# --------------------------------------------------------------
# NT Publisher Loop
# --------------------------------------------------------------
async def nt_loop():
    global last_pose
    while True:
        if last_pose is not None:
            print(last_pose)
            pos = last_pose[:3, 3]
            rot = Rotation.from_matrix(last_pose[:3, :3]).as_euler("xyz", degrees=True)
            camera_table.putNumberArray(
                "robotPoseField", [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]]
            )
            camera_table.putNumber("robotYawField", rot[2])
            last_pose = None
        else:
            camera_table.putNumberArray("robotPoseField", [0] * 6)
        camera_table.putNumber("latency", last_latency)
        await asyncio.sleep(0.05)


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
async def main():
    # global current_calibration, calib_config, nt_server_addr
    # global driver_mode
    global tag_size_m, current_cap, detector, field_tag_poses
    load_config()
    discover_cameras()
    open_selected_camera()
    apply_camera_settings()
    load_field_layout()
    init_detector()
    init_calib_board()

    if app_config.global_data.nt_server_addr:
        nt.initialize(server=str(app_config.global_data.nt_server_addr))
    else:
        nt.initialize()

    threading.Thread(target=video_loop, daemon=True).start()

    app = web.Application()
    app.router.add_get("/stream", mjpeg_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/", path=Path("www"), show_index=True)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    log.info("Server running on http://<ip>:8080")

    await nt_loop()


if __name__ == "__main__":
    asyncio.run(main())
