import cv2
from aiologger import Logger
from aiologger.levels import LogLevel
from cv2_enumerate_cameras import enumerate_cameras

from model import CameraData, UISettings, VisionSegment

log = Logger.with_default_handlers(name="photonvision_camera", level=LogLevel.INFO)


def discover_cameras(camera_state: VisionSegment):
    all_cams = enumerate_cameras()
    cams = []
    for cam in all_cams:
        cap = cv2.VideoCapture(cam.index)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cams.append(CameraData(index=cam.index, name=cam.name, res_x=w, res_y=h))
            cap.release()
    camera_state.available_cameras = cams
    log.info(f"Found {len(cams)} cameras")


def open_selected_camera(app_config: UISettings, camera_state: VisionSegment):
    with camera_state.camera_lock:
        if camera_state.current_cap:
            camera_state.current_cap.release()
        cap = cv2.VideoCapture(app_config.selected_camera)
        try:
            camera_data = [
                x
                for x in camera_state.available_cameras
                if x.index == app_config.selected_camera
            ][0]
            res_x = camera_data.res_x
            res_y = camera_data.res_y
        except IndexError:
            res_x = 1920
            res_y = 1080

        print("setting", res_x, res_y)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_x)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_y)
            cap.set(cv2.CAP_PROP_FPS, 120)
            camera_state.current_cap = cap
            log.info(f"Opened camera {app_config.selected_camera}")
        else:
            log.error(f"Failed to open camera {app_config.selected_camera}")


def apply_camera_settings(app_config: UISettings, camera_state: VisionSegment):
    if not camera_state.current_cap:
        print("cannot apply")
        return
    s = app_config.camera_settings
    camera_state.current_cap.set(
        cv2.CAP_PROP_EXPOSURE, s.exposure if not s.auto_exposure else -1
    )
    camera_state.current_cap.set(cv2.CAP_PROP_GAIN, s.gain)
    camera_state.current_cap.set(cv2.CAP_PROP_BRIGHTNESS, s.brightness)
    camera_state.current_cap.set(
        cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
        s.white_balance if not s.auto_white_balance else -1,
    )
    print("setting cap res")
    print(s.res_x, s.res_y)
    camera_state.current_cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.res_x)
    camera_state.current_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.res_y)
    print("setting cap res")
