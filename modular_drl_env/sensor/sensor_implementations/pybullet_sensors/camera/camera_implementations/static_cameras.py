import pybullet as pyb
from typing import Union, List, Dict, TypedDict
from modular_drl_env.robot.robot_implementations.pybullet_robots.ur5 import UR5_Pybullet
from ..camera_utils import *
from ..camera import CameraBase_Pybullet, CameraArgs # to prevent circular imports the things within the package have to be imported using the relative path

__all__ = [
    'StaticFloatingCameraFollowEffector_Pybullet',
    'StaticFloatingCamera_Pybullet',
]

class StaticFloatingCameraFollowEffector_Pybullet(CameraBase_Pybullet):
    """
    floating camera at position, if target is None, the camera will follow the robot's effector.
    """

    def __init__(self, robot : UR5_Pybullet, position: List, target: List = None, camera_args : CameraArgs = None, name : str = 'default_floating', **kwargs):
        super().__init__(target= target, camera_args= camera_args, name= name, **kwargs)
        self.robot = robot
        self.pos = position

    def _adapt_to_environment(self):
        self.target = pyb.getLinkState(self.robot.object_id, self.robot.end_effector_link_id)[4]
        super()._adapt_to_environment()

    def get_data_for_logging(self) -> dict:
        """
        Track target because reasons
        """
        dic = super().get_data_for_logging()
        dic[self.output_name + '_target'] = self.target
        return dic


class StaticFloatingCamera_Pybullet(CameraBase_Pybullet):
    """
    floating camera at position, if target is None, the camera will follow the robot's effector.
    """

    def __init__(self, position: List, target: List, camera_args : CameraArgs = None, name : str = 'default_floating', **kwargs):
        super().__init__(position = position, target= target, camera_args= camera_args, name= name, **kwargs)

    def _adapt_to_environment(self):
        """
        Since there are no changes to the camara's parameters we can just skip updating it
        """
        pass
        # super()._adapt_to_environment()