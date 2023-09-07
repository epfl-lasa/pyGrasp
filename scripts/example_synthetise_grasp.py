"""An example on how to use grasp synthesis in this package
"""
import random

import pyGrasp.utils as pgu
from pyGrasp.robot_model import RobotModel
from pyGrasp.grasp_synthesiser import GraspSynthesizer


# Choose your example robot here
SELECTED_ROBOT = pgu.IIWA7_URDF_PATH  # Find all possible robot in the utils.py file


def main() -> None:
    """Gets the model of a robot and.
    """

    random.seed(0)  # For repeatability

    # Load urdf
    if SELECTED_ROBOT.folder.is_dir() and SELECTED_ROBOT.file_path.is_file():
        robot_model = RobotModel(SELECTED_ROBOT.folder, SELECTED_ROBOT.file_path)
        print("Loaded robot model:")
        print(robot_model)
    else:
        raise FileNotFoundError(f"URDF provided is not a valid file path: {SELECTED_ROBOT}")

    # Create reachable space
    gs = GraspSynthesizer(robot_model)
    gs.sythetize_grasp()


if __name__ == "__main__":
    main()
