"""An example on how to use oppositions spaces in this package
"""
import random

import pyGrasp.utils as pgu
from pyGrasp.robot_model import RobotModel
from pyGrasp.opposition_spaces import OppositionSpace


# Choose your example robot here
SELECTED_ROBOT = pgu.IIWA7_URDF_PATH  # Find all possible robot in the utils.py file


def main() -> None:
    """Gets the model of a robot and then compute and displays the opposition spaces.
    Opposition are displayed sequentially in different windows.
    You need to close the previous one ot see the next one.
    Last window displays the opposition matrix for a given object diameter.
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
    opp_s = OppositionSpace(robot_model)
    opp_s.compute_os(force_recompute=False)

    # Show all os's to check
    opp_s.show_all_os(max_plots=30)
    opp_s.show_os_matrix(obj_diameter=0.3)


if __name__ == "__main__":
    main()
