#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import wpilib
from aiohttp import WSMsgType, web
from aiologger import Logger
from aiologger.levels import LogLevel
from robotpy_apriltag import AprilTagField, AprilTagFieldLayout

from calibration import compute_calibration, init_calib_board
from camera import apply_camera_settings, discover_cameras, open_selected_camera
from detector import init_detector, process_frame
from model import DetectorState, NetworkState, SafeFile, UISettings, VisionSegment
from network import nt_loop

script_path = Path(__file__).parent

async_log = Logger.with_default_handlers(name="photonvision", level=LogLevel.INFO)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("photonvision")

parser = argparse.ArgumentParser(prog="PythonVision", description="PythonVision Args.")
parser.add_argument(
    "-i",
    "--instance",
    type=str,
    required=True,
    help="The instance name of the vision server to help differentiate.",
)
parser.add_argument(
    "-p", "--port", type=int, required=True, help="The IP to run the instance on."
)
parser.add_argument(
    "-c",
    "--config_file",
    type=Path,
    required=True,
    help="The path of the config file with the instance settings.",
    default=script_path / Path("photonvision_config.json"),
)
parser.add_argument(
    "-d",
    "--ip_file",
    type=Path,
    help="The path of the IP file where the IP is configured. Only applies where the instance is running as 'master'",
    default=script_path / Path("ip_suffix.txt"),
)
parser.add_argument(
    "-m",
    "--master",
    action="store_true",
    help="Whether or not it is operating as master. The master instance can control the IP and hostname.",
)
args = parser.parse_args()

CONFIG_FILE = SafeFile(
    args.config_file,
    load_func=UISettings.model_validate_json,
    save_func=lambda data, f: json.dump(data, f, indent=2),
)

IP_FILE = SafeFile(
    args.ip_file,
    load_func=lambda data: data,
    save_func=lambda data, f: f.write(data),
)
_save_lock = threading.Lock()
_save_timer: Optional[threading.Timer] = None

app_config = UISettings()
camera_state = VisionSegment()
nt_state = NetworkState.quick_create(
    camera_name=app_config.global_data.camera_name,
    team_number=app_config.global_data.team_number,
)
ws_clients = set()


def _debounced_save():
    global _save_timer
    with _save_lock:
        if _save_timer:
            _save_timer.cancel()
        _save_timer = threading.Timer(0.5, _do_save)
        _save_timer.start()


def _do_save():
    CONFIG_FILE.save_file(app_config.model_dump(exclude_none=True))

    if args.master:
        ip_data = (
            f"{app_config.global_data.ip_suffix}\n{app_config.global_data.mdns_name}"
        )
        IP_FILE.save_file(ip_data)
    else:
        log.info("Not running as master. Will not update IP.")
    log.info("Config saved")
    # except Exception as e:
    #     async_log.error(f"Save failed: {e}")


def load_config():
    global app_config
    global nt_state
    if not CONFIG_FILE.path.exists():
        async_log.info("No config – using defaults")
        _do_save()
        return
    try:
        app_config = CONFIG_FILE.load_file()
        init_calib_board(app_config.calib_config)
        nt_state = NetworkState.quick_create(
            camera_name=app_config.global_data.camera_name,
            team_number=app_config.global_data.team_number,
        )
        log.info("Config loaded, networktable link recreated")
    except Exception as e:
        log.error(f"Config load failed: {e}")


detector_state = DetectorState()

# --------------------------------------------------------------
# Field Layout
# --------------------------------------------------------------
field_layout = AprilTagFieldLayout.loadField(AprilTagField.k2025ReefscapeWelded)


# --------------------------------------------------------------
# Video Capture Loop
# --------------------------------------------------------------
def video_loop():
    while True:
        if not camera_state.current_cap or not camera_state.current_cap.isOpened():
            log.warning(
                f"Could not acquire lock on selected camera {app_config.selected_camera}. Sleeping and re-trying."
            )
            if not camera_state.camera_lock.locked():
                log.info(
                    "Trying to open camera again since we are unable to acquire a lock and are not actively trying to connect/pull from the camera."
                )
                open_selected_camera(app_config, camera_state)
            time.sleep(0.25)
            continue
        with camera_state.camera_lock:
            ret, frame = camera_state.current_cap.read()
            capture_timestamp = int(wpilib.Timer.getFPGATimestamp() * 1e6)
        if not ret:
            if not camera_state.camera_lock.locked():
                log.info(
                    "Trying to open camera again since we have received a blank frame."
                )
                open_selected_camera(app_config, camera_state)
            continue
        try:
            process_frame(
                frame,
                app_config,
                detector_state,
                camera_state,
                field_layout,
                capture_timestamp,
                draw_ui_elements=not app_config.global_data.driver_mode,
            )
        except Exception as e:
            log.error(e)
            # Reset the last pose so that Robot code knows that we have stale data.
            camera_state.last_pnp = None
        finally:
            camera_state.current_frame_raw = frame
            camera_state.sequence_id += 1


async def mjpeg_handler(request: web.Request):
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


def make_config():
    async_log.info(app_config.model_dump())
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
                discover_cameras(camera_state)
                await ws.send_json(
                    {
                        "type": "cameras",
                        "list": camera_state.model_dump(include={"available_cameras"})[
                            "available_cameras"
                        ],
                    }
                )

            elif t == "get_instance":
                await ws.send_json({"instance": args.instance, "master": args.master})
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
                    update=data.get("camera_settings", {})
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
                init_calib_board(app_config.calib_config)

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
                nt_state.nt_instance.setServerTeam(app_config.global_data.team_number)

            elif t == "global":
                update_data = data.get("global", {})

                # Check if the camera name is updated
                camera_name_updated = (
                    "camera_name" in update_data
                    and update_data["camera_name"] != app_config.global_data.camera_name
                )
                app_config.global_data = app_config.global_data.model_copy(
                    update=data.get("global", {})
                )

                # Force the networktable path to be updated
                if camera_name_updated:
                    nt_state.update_camera_table(app_config.global_data.camera_name)
            _debounced_save()

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


async def main():
    load_config()
    discover_cameras(camera_state)
    open_selected_camera(app_config, camera_state)
    apply_camera_settings(app_config, camera_state)
    detector_state.detector = init_detector(app_config.pipeline)
    init_calib_board(app_config.calib_config)

    threading.Thread(target=video_loop, daemon=True).start()

    app = web.Application()
    app.router.add_get("/stream", mjpeg_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/", path=script_path / Path("www"), show_index=True)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    async_log.info("Server running on http://<ip>:%s" % args.port)

    await nt_loop(camera_state, nt_state)
    # threading.Thread(target=video_loop, daemon=True).start()
    # await video_loop()


if __name__ == "__main__":
    asyncio.run(main())
