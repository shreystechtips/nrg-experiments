import traceback

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


def compute_calibration(app_config, logger: Logger):
    captures = app_config.calib_config.calib_runtime.captures
    if len(captures) < 10:
        return "Need at least 10 good frames for reliable Charuco calibration"

    total_object_points = []
    total_image_points = []
    shape: tuple[int, int] | None = None
    for i, im in enumerate(captures):
        logger.info("=> Processing image {0}".format(i))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if shape is None:
            shape = (gray.shape[0], gray.shape[1])
        else:
            if shape != (gray.shape[0], gray.shape[1]):
                logger.info(im, "index was not the same size!")
                return "index was not the same size!"
        detector = cv2.aruco.CharucoDetector(
            app_config.calib_config.calib_runtime.calib_board
        )
        corners, ids, aruco_corners, aruco_ids = detector.detectBoard(gray)
        if ids is None or (len(ids)) <= 0:
            logger.info("Not enough IDs for calib")
            continue

        pt_ref = app_config.calib_config.calib_runtime.calib_board.matchImagePoints(
            corners, ids
        )
        if len(pt_ref[0]) < 4:
            logger.info("Not enough points for calibration")
            continue
        total_object_points.append(pt_ref[0])
        total_image_points.append(pt_ref[1])

    res_key = f"{shape[1]}x{shape[0]}"
    try:
        rms, m, d, _r, _t = cv2.calibrateCamera(
            total_object_points, total_image_points, shape, None, None
        )
        if app_config.selected_camera not in app_config.calibration:
            app_config.calibration[app_config.selected_camera] = dict()
        app_config.calibration[app_config.selected_camera][res_key] = CalibrationData(
            cameraMatrix=m.tolist(), distCoeffs=d.tolist()
        )
    except:
        return f"Error: {traceback.format_exc()}"
    log.info(f"Calibration done: RMS={rms:.3f}")
    app_config.calib_config.calib_runtime.captures.clear()
    return f"Matrix: {m.tolist()}\nCoeffs: {d.tolist()}\nRMS error: {rms:.3f}"
