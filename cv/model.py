import threading
import time

import cv2
import numpy as np
from ntcore import NetworkTable, NetworkTableInstance
from photonlibpy.networktables.NTTopicSet import NTTopicSet
from pupil_apriltags import Detector
from pydantic import BaseModel, ConfigDict, Field, computed_field


class CameraSettings(BaseModel):
    exposure: int = 12000
    gain: int = 12
    brightness: int = 50
    white_balance: int = 4600
    auto_exposure: bool = False
    auto_white_balance: bool = True
    led_mode: int = 1
    res_x: int = 1920
    res_y: int = 1080


class PipelineSettings(BaseModel):
    family: str = "tag36h11"
    nthreads: int = 4
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.8


class CalibrationData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cameraMatrix: np.ndarray
    distCoeffs: np.ndarray
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
    camera_name: str = Field(default="change_me")


class CalibrationCaptures(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    captures: list[np.ndarray] = Field(default=[])
    calib_board: cv2.aruco.GridBoard | cv2.aruco.CharucoBoard | None = Field(
        default=None
    )


class CalibConfig(BaseModel):
    board_type: str = Field(default="Charuco")
    tag_family: str = Field(default="DICT_4X4_1000")
    pattern_spacing_in: float = Field(default=1.0)
    marker_size_in: float = Field(default=0.75)
    board_width_sq: int = Field(default=8)
    board_height_sq: int = Field(default=8)
    calib_runtime: CalibrationCaptures = Field(
        default_factory=CalibrationCaptures, exclude=True
    )


class UISettings(BaseModel):
    selected_camera: int = Field(default=0)
    camera_settings: CameraSettings = Field(default_factory=CameraSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
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
    last_pose: np.ndarray | None = None
    last_latency: float = 0.0
    last_ids: list[int] = Field(default_factory=list)
    last_capture_time: int = -1
    sequence_id: int = 0

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

    @classmethod
    def quick_create(
        cls,
        camera_name: str,
        team_number: int,
        table_name: str = "photonvision",
    ) -> "NetworkState":
        nt_instance = NetworkTableInstance.getDefault()
        nt_instance.startClient3("photonvision")
        nt_instance.setServerTeam(team_number)
        # For local testing
        # nt_instance.setServer("localhost", NetworkTableInstance.kDefaultPort3)
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
