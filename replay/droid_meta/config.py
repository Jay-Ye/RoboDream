import os

import numpy as np

CAM2IMAGE_MAP = {
    "ext1_cam_serial": "exterior_image_1_left",
    "ext2_cam_serial": "exterior_image_2_left",
    "wrist_cam_serial": "wrist_image_left",
}
IMAGE2SCENE_CAM_MAP = {
    "exterior_image_1_left": "external_cam1",
    "exterior_image_2_left": "external_cam2",
    "wrist_image_left": "wrist_cam",
}
SCENE_CAM2IMAGE_MAP = {
    "external_cam1": "exterior_image_1_left",
    "external_cam2": "exterior_image_2_left",
    "wrist_cam": "wrist_image_left",
}
DEFAULT_CAMERA_INTRINSICS = np.array([[525.,   0.,         648.12060547],
                                    [  0.,         525., 374.60479736],
                                    [  0.,           0.,           1.        ]])
DEFAULT_CAMERA_INTRINSICS_WRIST = np.array([[732., 0., 640],
                                            [0., 732., 360],
                                            [0., 0., 1.]])

_DEFAULT_CAMERA_INTRINSICS = [DEFAULT_CAMERA_INTRINSICS[0, 0], DEFAULT_CAMERA_INTRINSICS[0, 2], DEFAULT_CAMERA_INTRINSICS[1, 1], DEFAULT_CAMERA_INTRINSICS[1, 2]]
_DEFAULT_CAMERA_INTRINSICS_WRIST = [DEFAULT_CAMERA_INTRINSICS_WRIST[0, 0], DEFAULT_CAMERA_INTRINSICS_WRIST[0, 2], DEFAULT_CAMERA_INTRINSICS_WRIST[1, 1], DEFAULT_CAMERA_INTRINSICS_WRIST[1, 2]]

# Data locations (see .env.example / shell/export.sh)
DATA_DIR = os.environ.get("ROBODREAM_DATA_DIR", "datasets")
DROID_META_DIR = os.environ.get("DROID_META_DIR", os.path.join(DATA_DIR, "droid_meta"))
local_dataset_path = os.environ.get("DROID_ROOT", "droid")  # raw DROID release (contains <lab>/success/<date>/<episode>/trajectory.h5)
replay_img_save_path = os.environ.get("ROBODREAM_RENDER_DIR", os.path.join(DATA_DIR, "droid_renders"))  # robot-only renders

def _format_intrinsics(intrinsics):
    reformatted_intrinsics = {
        "cameraMatrix": intrinsics,
        "distCoeffs": [0.0] * 12,
    }
    if intrinsics[0] > 500 or intrinsics[0] > 700:
        reformatted_intrinsics.update({
            "width": 1280,
            "height": 720,
        })
    else:
        reformatted_intrinsics.update({
            "width": 672,
            "height": 376,
        })
    return reformatted_intrinsics

def format_intrinsics(intrinsics: list[float], is_wrist: bool):
    if sum(intrinsics) == 0: # no intrinsics available, use default intrinsics
        return _format_intrinsics(_DEFAULT_CAMERA_INTRINSICS_WRIST if is_wrist else _DEFAULT_CAMERA_INTRINSICS)
    return _format_intrinsics(intrinsics)