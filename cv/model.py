import logging
import shutil
import threading
import time
from pathlib import Path
from traceback import format_exc

import cv2
import numpy as np
from ntcore import NetworkTable, NetworkTableInstance
from photonlibpy.networktables.NTTopicSet import NTTopicSet
from photonlibpy.targeting.multiTargetPNPResult import PnpResult
from photonlibpy.targeting.photonTrackedTarget import PhotonTrackedTarget
from pupil_apriltags import Detector
from pydantic import BaseModel, ConfigDict, Field, computed_field

log = logging.getLogger("photonvision_model")


class CameraSettings(BaseModel):
    exposure: int = 500
    gain: int = 2
    brightness: int = 2
    white_balance: int = 4600
    auto_exposure: bool = True
    auto_white_balance: bool = True
    led_mode: int = 1
    res_x: int = 1280
    res_y: int = 800


class PipelineSettings(BaseModel):
    family: str = "tag36h11"
    nthreads: int = 8
    quad_decimate: float = 2.0
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.25


class CalibrationData(BaseModel):
    cameraMatrix: list[list[float]]
    distCoeffs: list[list[float]]


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
    led_brightness: int = Field(default=0, ge=0, le=1)
    driver_mode: bool = Field(default=False)
    camera_name: str = Field(default="change_me")
    ip_suffix: int = Field(default=11, gt=10, le=254)
    mdns_name: str = Field(default="default-camera")


class CalibrationCaptures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    captures: list[np.ndarray] = Field(default=[])
    calib_board: cv2.aruco.GridBoard | cv2.aruco.CharucoBoard | None = Field(
        default=None
    )


class CalibConfig(BaseModel):
    board_type: str = Field(default="Charuco")
    tag_family: str = Field(default="DICT_4X4_1000")
    sq_len: float = Field(default=1)
    marker_len: float = Field(default=0.75)
    board_width_sq: int = Field(default=8)
    board_height_sq: int = Field(default=8)
    calib_runtime: CalibrationCaptures = Field(
        default_factory=CalibrationCaptures, exclude=True
    )


class UISettings(BaseModel):
    selected_camera: int = Field(default=0)
    camera_settings: CameraSettings = Field(default_factory=CameraSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    calibration: dict[int, dict[str, CalibrationData]] = Field(default_factory=dict)
    robot_offset: RobotOffset = Field(default_factory=RobotOffset)
    global_data: GlobalSettings = Field(default_factory=GlobalSettings)
    calib_config: CalibConfig = Field(default_factory=CalibConfig)


class CameraData(BaseModel):
    index: int
    name: str
    res_x: int
    res_y: int

    @computed_field
    def res(self) -> str:
        return f"{self.res_x}x{self.res_y}"


class VisionSegment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    available_cameras: list[CameraData] = Field(default=[])
    camera_lock: threading.Lock = Field(default_factory=threading.Lock)
    current_cap: cv2.VideoCapture | None = Field(default=None)
    current_frame_raw: np.ndarray | None = None
    current_frame_processed: np.ndarray | None = None
    last_pnp: PnpResult | None = None
    last_latency: float = 0.0
    last_targets: list[PhotonTrackedTarget] = Field(default_factory=list)
    last_capture_time: int = -1
    sequence_id: int = 0

    @computed_field
    @property
    def last_ids(self) -> list[int]:
        return [target.fiducialId for target in self.last_targets]

    @computed_field
    def fps(self) -> float:
        return 1 / (self.last_latency) if self.last_latency else 0


class DetectorState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    detector: Detector | None = None


class NetworkState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    root_table: NetworkTable
    cam_table: NetworkTable
    nt_wrapper: NTTopicSet
    nt_instance: NetworkTableInstance

    def update_camera_table(self, new_name: str):
        self.cam_table = self.root_table.getSubTable(new_name)
        self.nt_wrapper = NTTopicSet(self.cam_table)
        self.nt_wrapper.updateEntries()

    @classmethod
    def quick_create(
        cls,
        camera_name: str,
        team_number: int,
        table_name: str = "photonvision",
    ) -> "NetworkState":
        nt_instance = NetworkTableInstance.getDefault()
        nt_instance.startClient4("photonvision")
        nt_instance.setServerTeam(team_number)
        # For local testing
        # nt_instance.setServer("localhost", NetworkTableInstance.kDefaultPort4)
        # print("Waiting for connection...")
        # while not nt_instance.isConnected():
        #     time.sleep(0.1)
        # print("Connected!")
        root_table = nt_instance.getTable(table_name)
        cam_table = root_table.getSubTable(camera_name)
        nt_wrapper = NTTopicSet(cam_table)
        nt_wrapper.updateEntries()
        return cls(
            root_table=root_table,
            cam_table=cam_table,
            nt_instance=nt_instance,
            nt_wrapper=nt_wrapper,
        )


class SafeFile:
    """A file which tries to prevent memory corruption on saves."""

    def __init__(self, path: Path, load_func, save_func):
        self.path: Path = path
        self.save_func = save_func
        self.load_func = load_func
        self.tmp: Path = self.path.with_suffix(".tmp")

    def save_file(self, data, **kwargs):
        """Save a file safely to ensure that data is not corrupted on the main file."""
        # Save file to temp location.
        with self.tmp.open("w") as f:
            self.save_func(data, f, **kwargs)
        # Copy the file.
        shutil.copy(self.tmp, self.path)

        # Delete the file.
        self.tmp.unlink()

    def load_file(self):
        # Check if the temp file still exists from an unclean reboot.
        if self.tmp.exists():
            try:
                with open(self.tmp, "r") as f:
                    loaded_file = self.load_func(f.read())
                    self.tmp.unlink()
                    log.info("Loaded file from old half-save state")
                    return loaded_file
            except:
                log.error(
                    "Error loading file. Leaving it as is for debug \n %s", format_exc()
                )
        with open(self.path, "r") as f:
            return self.load_func(f.read())
