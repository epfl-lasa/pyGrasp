"""A class to instantiate a robot model and provides lots of methods through roboticstoolbox superclass
"""
import pathlib
import typing as tp
from roboticstoolbox.robot.ERobot import ERobot


class RobotModel(ERobot):
    """Child of ERobot, really just here to give access to all the methods of RTB with any URDF
    """

    def __init__(self, urdf_file: tp.Union[str, pathlib.Path]) -> None:

        (e_links, name, self._urdf, self._file_path) = ERobot.URDF_read(urdf_file)
        super().__init__(e_links, name=name)
