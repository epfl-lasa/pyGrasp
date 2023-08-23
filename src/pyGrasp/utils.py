"""This file contains a few definitions, folders and paths that are used through out
the pyGrasp module and make for more concise code in other modules.
"""
import os
from pathlib import Path
from collections import namedtuple



# Root of the pygrasp package in absolute
PYGRASP_ROOT = Path(os.path.dirname(__file__))

# Cache folder for pygrasp
CACHE_FOLDER = PYGRASP_ROOT / ".cache"
if not CACHE_FOLDER.is_dir():
    os.mkdir(str(CACHE_FOLDER))

# A class to hold the infos about a path to a URDF
UrdfPath = namedtuple("UrdfPath", ["folder", "file_path"])

# All available robot with their path descriptions
IIWA_FOLDER = PYGRASP_ROOT / "../../models/iiwa/"
ALLEGRO_FOLDER = PYGRASP_ROOT / "../../models/allegro/"

IIWA7_URDF_PATH = UrdfPath(
    folder=IIWA_FOLDER,
    file_path=IIWA_FOLDER / Path("iiwa_description/urdf/iiwa7.urdf.xacro")
)
IIWA14_URDF_PATH = UrdfPath(
    folder=IIWA_FOLDER,
    file_path=IIWA_FOLDER / Path("iiwa_description/urdf/iiwa14.urdf.xacro")
)
ALLEGRO_LEFT_URDF_PATH = UrdfPath(
    folder=ALLEGRO_FOLDER,
    file_path=ALLEGRO_FOLDER / Path("allegro_hand_description/allegro_hand_description_left.urdf")
)
ALLEGRO_RIGHT_URDF_PATH = UrdfPath(
    folder=ALLEGRO_FOLDER,
    file_path=ALLEGRO_FOLDER / Path("allegro_hand_description/allegro_hand_description_right.urdf"))
