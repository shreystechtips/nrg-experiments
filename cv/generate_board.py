from pydantic import BaseModel, Field
import cv2
import sys
from calibration import init_calib_board
from model import CalibConfig


cfg = CalibConfig()
b = init_calib_board(cfg)
print(f"Generating board image with params", cfg.dict(), "and length 2000px")
image_name = f"{sys.argv[1]}.png"
ratio = cfg.board_height_sq / cfg.board_width_sq
charuco_board_image = b.generateImage((2000, int(2000 * ratio)), marginSize=20)
cv2.imwrite(image_name, charuco_board_image)
