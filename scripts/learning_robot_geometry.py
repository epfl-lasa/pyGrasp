"""This file contains an example of a robot learning the geometry of its links through gpr"""


from pathlib import Path
import os
import numpy as np
import random

from pyGrasp.robot_model import RobotModel


def main() -> None:
    """Gets the model of the IIWA, print it and performs FK to a random link. 
    """

    random.seed(0)  # For repeatability

    # Find an example URDF (iiwa7)
    here = os.path.dirname(__file__)
    urdf_folder = Path(here) / Path("../models/iiwa/")
    urdf_path = urdf_folder / Path("iiwa_description/urdf/iiwa7.urdf.xacro")

    # Load urdf
    if urdf_folder.is_dir() and urdf_path.is_file():
        robot_model = RobotModel(urdf_folder, urdf_path)
        print("Loaded robot model:")
        print(robot_model)
    else:
        raise FileNotFoundError(f"URDF provided is not a valid file path: {urdf_path}")

    # Learning robot geometry and plot
    robot_model.learn_geometry(nb_learning_pts=10000, verbose=True)
    robot_model.show_geometries()


if __name__ == "__main__":
    main()
