import cv2
from aiologger import Logger
from aiologger.levels import LogLevel

from model import CalibConfig, CalibrationData

log = Logger.with_default_handlers(name="photonvision_calibration", level=LogLevel.INFO)


def init_calib_board(cfg: CalibConfig):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.tag_family))
    board = cv2.aruco.CharucoBoard(
        (cfg.board_width_sq, cfg.board_height_sq),
        cfg.sq_len * 0.0254,  # in → m,
        cfg.marker_len * 0.0254,  # in → m,
        dictionary,
    )
    cfg.calib_runtime.calib_board = board
    return board


def compute_calibration(app_config):
    captures = app_config.calib_config.calib_runtime.captures
    if len(captures) < 10:
        print("Need at least 10 good frames for reliable Charuco calibration")
        return False

    total_object_points = []
    total_image_points = []
    shape: tuple[int, int] | None = None
    for i, im in enumerate(captures):
        print("=> Processing image {0}".format(i))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if shape is None:
            shape = (gray.shape[0], gray.shape[1])
        else:
            if shape != (gray.shape[0], gray.shape[1]):
                print(im, "index was not the same size!")
                return False
        detector = cv2.aruco.CharucoDetector(
            app_config.calib_config.calib_runtime.calib_board
        )
        corners, ids, aruco_corners, aruco_ids = detector.detectBoard(gray)
        if ids is None or (len(ids)) <= 0:
            print("Not enough IDs for calib")
            continue

        pt_ref = app_config.calib_config.calib_runtime.calib_board.matchImagePoints(
            corners, ids
        )
        if len(pt_ref[0]) < 4:
            print("Not enough points for calibration")
            continue
        total_object_points.append(pt_ref[0])
        total_image_points.append(pt_ref[1])

    res_key = f"{shape[1]}x{shape[0]}"
    rms, m, d, _r, _t = cv2.calibrateCamera(
        total_object_points, total_image_points, shape, None, None
    )
    if app_config.selected_camera not in app_config.calibration:
        app_config.calibration[app_config.selected_camera] = dict()
    app_config.calibration[app_config.selected_camera][res_key] = CalibrationData(
        cameraMatrix=m.tolist(), distCoeffs=d.tolist()
    )
    log.info(f"Calibration done: RMS={rms:.3f}")
    app_config.calib_config.calib_runtime.captures.clear()
    return True
