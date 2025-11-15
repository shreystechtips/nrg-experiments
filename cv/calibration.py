import logging

import cv2
import numpy as np

from model import CalibrationData

log = logging.getLogger("photonvision_calibration")


def init_calib_board(app_config):
    cfg = app_config.calib_config
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
    app_config.calib_config.calib_runtime.calib_board = board


def compute_calibration(app_config):
    if len(app_config.calib_config.calib_runtime.captures) < 10:
        log.warning("Need at least 10 good frames")
        return False

    res = app_config.calib_config.calib_runtime.captures[0].shape[1::-1]  # (w, h)
    res_key = f"{res[0]}x{res[1]}"

    all_charuco_corners = []
    all_charuco_ids = []
    all_gray = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        for f in app_config.calib_config.calib_runtime.captures
    ]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for gray in all_gray:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, app_config.calib_config.calib_runtime.calib_board.getDictionary()
        )
        if len(corners) == 0:
            continue
        if isinstance(
            app_config.calib_config.calib_runtime.calib_board, cv2.aruco.CharucoBoard
        ):
            n, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, app_config.calib_config.calib_runtime.calib_board
            )
            if charuco_ids is not None and len(charuco_ids) > 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
        else:
            obj_points, img_points = (
                app_config.calib_config.calib_runtime.calib_board.getGridPoints(
                    corners, ids
                )
            )
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
        app_config.calib_config.calib_runtime.calib_board,
        (w, h),
        K,
        dist,
        criteria=criteria,
    )

    app_config.calibration.size_calib_data[res_key] = CalibrationData(
        cameraMatrix=K.tolist(),
        distCoeffs=dist.tolist(),
        reprojectionError=float(rms),
    )
    log.info(f"Calibration done: RMS={rms:.3f}")
    app_config.calib_config.calib_runtime.captures.clear()
    return True
