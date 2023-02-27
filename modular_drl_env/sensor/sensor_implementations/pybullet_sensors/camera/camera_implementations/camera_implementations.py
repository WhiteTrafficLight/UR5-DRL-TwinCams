import pybullet as pyb
from typing import List
from modular_drl_env.robot.robot_implementations.ur5 import UR5
from ..camera_utils import *
from ..camera import CameraBase_Pybullet

__all__ = [
    'StaticBodyCameraUR5_Pybullet',
    'StaticFloatingCamera_Pybullet'
]

class StaticBodyCameraUR5_Pybullet(CameraBase_Pybullet):

    def __init__(self, robot : UR5, position_relative_to_effector: List = None, camera_args: dict = None, name : str = 'default_body_ur5', **kwargs):
        self.robot = robot
        self.relative_pos = position_relative_to_effector
        super().__init__(camera_args= camera_args, name= name, **kwargs)

    def _calculate_position(self):
        effector_position, effector_orientation = pyb.getLinkState(self.robot.object_id, self.robot.end_effector_link_id)[4:6]
        body_position, body_orientation = pyb.getLinkState(self.robot.object_id, self.robot.end_effector_link_id - 1)[4:6]
        effector_up_vector, effector_forward_vector, _ = directionalVectorsFromQuaternion(effector_orientation)
        self.camera_args['up_vector'] = effector_up_vector
        if self.relative_pos is None:
            target = add_list(effector_position, effector_forward_vector) # [p+v for p,v in zip(effector_position, effector_forward_vector)]
            body_forward_vector, body_up_vector, _ = directionalVectorsFromQuaternion(body_orientation)
            position = add_list(add_list(body_position, body_up_vector, 0.075), body_forward_vector, 0.075) # [p+u+f for p,u,f in zip(body_position, body_up_vector, body_forward_vector)]
        else:
            position = add_list(effector_position, self.relative_pos)
            target = add_list(position, effector_forward_vector)
        
        return position, target

    def _adapt_to_environment(self):
        self.pos, self.target = self._calculate_position()
        super()._adapt_to_environment()

    def get_data_for_logging(self) -> dict:
        """
        Also track position because it moves
        """
        dic = super().get_data_for_logging()
        dic[self.output_name + '_pos'] = self.pos
        return dic

class StaticFloatingCameraFollowEffector_Pybullet(CameraBase_Pybullet):
    """
    floating camera at position, if target is None, the camera will follow the robot's effector.
    """

    def __init__(self, robot : UR5, position: List, target: List = None, camera_args: dict = None, name : str = 'default_floating', **kwargs):
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

    def __init__(self, position: List, target: List, camera_args: dict = None, name : str = 'default_floating', **kwargs):
        super().__init__(position = position, target= target, camera_args= camera_args, name= name, **kwargs)

    def _adapt_to_environment(self):
        """
        Since there are no changes to the camara's parameters we can just skip updating it
        """
        pass
        # super()._adapt_to_environment()
