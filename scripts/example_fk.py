"""An example on how to use simple FK form the RobotModel in this package
"""
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

    # Select random joints and angles for FK
    q_fk = np.array([random.uniform(q_min, q_max) for (q_min, q_max) in robot_model.qlim.transpose()])
    link_origin_id = random.randint(0, robot_model.nlinks-1)
    link_goal_id = random.randint(link_origin_id + 1, robot_model.nlinks)
    link_origin = robot_model.links[link_origin_id]
    link_goal = robot_model.links[link_goal_id]

    # Perform FK
    fk_result = robot_model.fkine(q_fk, link_goal, link_origin)
    fk_jacob = robot_model.jacobe(q_fk, link_goal, link_origin)

    # Display results
    print(f"Result for forward kinematic\n"
          f"Origin link: {robot_model.links[link_origin_id].name}\n"
          f"Goal  link: {robot_model.links[link_goal_id].name}\n"
          f"Joint position: {q_fk}\n\n")
    print("SE3 transform matrix:\n", fk_result)

    print("Geometrical jacobian: \n", fk_jacob)
    
    # Plot FK
    robot_model.plot(q_fk, backend='swift', block=True)


if __name__ == "__main__":
    main()
