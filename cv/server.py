#!/usr/bin/env python3
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
from aiohttp import WSMsgType, web
from networktables import NetworkTables
from scipy.spatial.transform import Rotation

from calibration import compute_calibration, init_calib_board
from camera import apply_camera_settings, discover_cameras, open_selected_camera
from detector import init_detector, process_frame
from model import (
    DetectorState,
    UISettings,
    VisionSegment,
)

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


app_config = UISettings()
camera_state = VisionSegment()
nt = NetworkTables
camera_table = nt.getTable("/photonvision/default")


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
        json.dump(app_config.model_dump(exclude_none=True), f, indent=2)
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
            # data = json.load(f)
            # print(data)
            app_config = UISettings.model_validate_json(f.read())
        init_calib_board(app_config)
        log.info("Config loaded")
    except Exception as e:
        log.error(f"Config load failed: {e}")


detector_state = DetectorState()

# --------------------------------------------------------------
# Field Layout
# --------------------------------------------------------------
FIELD_LAYOUT_URL = "https://raw.githubusercontent.com/wpilibsuite/allwpilib/main/apriltag/src/main/native/resources/edu/wpi/first/apriltag/2025-reefscape-welded.json"
field_tag_poses: Dict[int, Dict] = {}


def load_field_layout():
    global field_tag_poses
    try:
        import requests

        # r = requests.get(FIELD_LAYOUT_URL, timeout=5)
        # r.raise_for_status()
        # data = r.json()
        with open("2025-reefscape-welded.json", "r") as f:
            data = json.load(f)
        field_tag_poses = {t["ID"]: t["pose"] for t in data.get("tags", [])}
        log.info(f"Loaded {len(field_tag_poses)} tags")
    except Exception as e:
        log.warning(f"Field layout failed: {e}")


# --------------------------------------------------------------
# Video Capture Loop
# --------------------------------------------------------------
def video_loop():
    while True:
        if not camera_state.current_cap or not camera_state.current_cap.isOpened():
            time.sleep(0.1)
            continue
        ret, frame = camera_state.current_cap.read()
        if not ret:
            continue
        camera_state.current_frame_raw = frame.copy()
        if not app_config.global_data.driver_mode:
            process_frame(
                frame, app_config, detector_state, camera_state, field_tag_poses
            )
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
            camera_state.current_frame_processed
            if camera_state.current_frame_processed is not None
            else camera_state.current_frame_raw
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
        ).encode("utf-8")  # <-- pure bytes

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
                await ws.send_json(
                    {
                        "type": "cameras",
                        "list": camera_state.model_dump(include={"available_cameras"})[
                            "available_cameras"
                        ],
                    }
                )

            elif t == "get_config":
                await ws.send_json(make_config())

            elif t == "select_camera":
                idx = data.get("index", 0)
                if any(c.index == idx for c in camera_state.available_cameras):
                    app_config.selected_camera = idx
                    open_selected_camera(app_config, camera_state)
                    apply_camera_settings(app_config, camera_state)
                    _debounced_save()
                    await broadcast({"type": "camera_selected", "index": idx})

            elif t == "config":
                app_config.camera_settings = app_config.camera_settings.model_copy(
                    update=data.get("camera_settinfs", {})
                )
                apply_camera_settings(app_config, camera_state)

            elif t == "pipeline":
                app_config.pipeline = app_config.pipeline.model_copy(
                    update=data.get("settings", {})
                )
                detector_state.detector = init_detector(app_config.pipeline)

            elif t == "calib_start":
                # TODO: calibration routine
                pass

            elif t == "calib_config":
                app_config.calib_config = app_config.calib_config.model_copy(
                    update=data.get("config", {})
                )
                init_calib_board(app_config)

            elif t == "calib_capture":
                if camera_state.current_frame_raw is not None:
                    app_config.calib_config.calib_runtime.captures.append(
                        camera_state.current_frame_raw.copy()
                    )
                    await ws.send_json(
                        {
                            "type": "calib_status",
                            "count": len(
                                app_config.calib_config.calib_runtime.captures
                            ),
                        }
                    )

            elif t == "calib_compute":
                success = compute_calibration(app_config)
                await ws.send_json({"type": "calib_result", "success": success})

            elif t == "calib_clear":
                app_config.calib_config.calib_runtime.captures.clear()
                await ws.send_json({"type": "calib_status", "count": 0})

            elif t == "nt_server":
                # global nt_server_addr
                app_config.global_data.nt_server_addr = data.get("address", "")
                nt.shutdown()
                if app_config.global_data.nt_server_addr:
                    nt.initialize(server=app_config.global_data.nt_server_addr)
                else:
                    nt.initialize()

            elif t == "global":
                app_config.global_data = app_config.global_data.model_copy(
                    update=data.get("global", {})
                )

    finally:
        _debounced_save()
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
    while True:
        if camera_state.last_pose is not None:
            print(camera_state.last_pose)
            pos = camera_state.last_pose[:3, 3]
            rot = Rotation.from_matrix(camera_state.last_pose[:3, :3]).as_euler(
                "xyz", degrees=True
            )
            camera_table.putNumberArray(
                "robotPoseField", [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]]
            )
            camera_table.putNumber("robotYawField", rot[2])
            camera_state.last_pose = None
        else:
            camera_table.putNumberArray("robotPoseField", [0] * 6)
        camera_table.putNumber("latency", camera_state.last_latency)
        await asyncio.sleep(0.05)


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
async def main():
    # global driver_mode
    global detector, field_tag_poses
    load_config()
    discover_cameras(camera_state)
    open_selected_camera(app_config, camera_state)
    apply_camera_settings(app_config, camera_state)
    load_field_layout()
    detector_state.detector = init_detector(app_config.pipeline)
    init_calib_board(app_config)

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
