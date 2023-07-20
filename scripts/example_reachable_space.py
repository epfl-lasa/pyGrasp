"""An example on how to use simple FK form the RobotModel in this package
"""
from pathlib import Path
import os
import random
from collections import namedtuple

from pyGrasp.robot_model import RobotModel
from pyGrasp.reachable_spaces import ReachableSpaces


UrdfPath = namedtuple("UrdfPath", ["folder", "file_path"])


# All availabel robot with their path descriptions
IIWA7_URDF_PATH = UrdfPath(folder=Path("../models/iiwa/"),
                           file_path=Path("iiwa_description/urdf/iiwa7.urdf.xacro"))
IIWA14_URDF_PATH = UrdfPath(folder=Path("../models/iiwa/"),
                            file_path=Path("iiwa_description/urdf/iiwa14.urdf.xacro"))
ALLEGRO_LEFT_URDF_PATH = UrdfPath(folder=Path("../models/allegro/"),
                                  file_path=Path("allegro_hand_description/allegro_hand_description_left.urdf"))
ALLEGRO_RIGHT_URDF_PATH = UrdfPath(folder=Path("../models/allegro/"),
                                   file_path=Path("allegro_hand_description/allegro_hand_description_right.urdf"))

# Choose your example robot here
SELECTED_ROBOT = ALLEGRO_LEFT_URDF_PATH


def main() -> None:
    """Gets the model of the IIWA, print it and performs FK to a random link. 
    """

    random.seed(0)  # For repeatability

    # Find an example URDF (iiwa7)
    here = os.path.dirname(__file__)
    urdf_folder = Path(here) / SELECTED_ROBOT.folder
    urdf_path = urdf_folder / SELECTED_ROBOT.file_path

    # Load urdf
    if urdf_folder.is_dir() and urdf_path.is_file():
        robot_model = RobotModel(urdf_folder, urdf_path)
        print("Loaded robot model:")
        print(robot_model)
    else:
        raise FileNotFoundError(f"URDF provided is not a valid file path: {urdf_path}")
    
    # Create reachable space
    reachable_space = ReachableSpaces(robot_model)
    reachable_space.compute()
    breakpoint()


if __name__ == "__main__":
    main()