import logging

import cv2
from aiologger import Logger
from aiologger.levels import LogLevel
from cv2_enumerate_cameras import enumerate_cameras

from model import CameraData, UISettings, VisionSegment

async_log = Logger.with_default_handlers(
    name="photonvision_camera", level=LogLevel.INFO
)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("photonvision_camera")


def discover_cameras(camera_state: VisionSegment):
    all_cams = enumerate_cameras()
    cams = []
    for cam in all_cams:
        try:
            cap = cv2.VideoCapture(cam.index)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cams.append(
                    CameraData(index=cam.index, name=cam.name, res_x=w, res_y=h)
                )
                cap.release()
        except:
            continue
    camera_state.available_cameras = cams
    log.info(f"Found {len(cams)} cameras")


def open_selected_camera(app_config: UISettings, camera_state: VisionSegment):
    with camera_state.camera_lock:
        if camera_state.current_cap:
            log.info("Releasing camera stream")
            camera_state.current_cap.release()
        try:
            cap = cv2.VideoCapture(app_config.selected_camera)
            if cap.isOpened():
                camera_state.current_cap = cap
                log.info(f"Opened camera {app_config.selected_camera}")
                apply_camera_settings(app_config, camera_state, create_lock=False)
            else:
                log.error(f"Failed to open camera {app_config.selected_camera}")
        except Exception as e:
            log.error(e)


def apply_camera_settings(
    app_config: UISettings, camera_state: VisionSegment, create_lock: bool = True
):
    if not camera_state.current_cap:
        log.info("Cannot apply settings as there is no camera open")
        return
    s = app_config.camera_settings
    if create_lock:
        camera_state.camera_lock.acquire()
    try:
        camera_state.current_cap.set(
            cv2.CAP_PROP_EXPOSURE, s.exposure if not s.auto_exposure else -1
        )
        camera_state.current_cap.set(cv2.CAP_PROP_GAIN, s.gain)
        camera_state.current_cap.set(cv2.CAP_PROP_BRIGHTNESS, s.brightness)
        camera_state.current_cap.set(
            cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
            s.white_balance if not s.auto_white_balance else -1,
        )
        log.info("Setting cap res %d x %d", s.res_x, s.res_y)
        camera_state.current_cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.res_x)
        camera_state.current_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.res_y)
        camera_state.current_cap.set(cv2.CAP_PROP_FPS, 120)
    finally:
        if create_lock:
            camera_state.camera_lock.release()
