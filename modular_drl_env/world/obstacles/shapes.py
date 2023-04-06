from modular_drl_env.world.obstacles.obstacle import Obstacle
import numpy as np
from typing import Union
from modular_drl_env.util.pybullet_util import pybullet_util as pyb_u

class Sphere(Obstacle):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, move_step: float, radius: float, color: list=[0.75,0.75,0.75,1]) -> None:
        super().__init__(position, rotation, trajectory, move_step)

        self.radius = radius
        self.color = color

    def build(self) -> int:
        self.object_id = pyb_u.create_sphere(position=self.position_orig,
                                             mass=0,
                                             radius=self.radius,
                                             color=self.color)
        return self.object_id

class Box(Obstacle):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, move_step: float, halfExtents=[1, 1, 1], color=[0.5,0.5,0.5,1]) -> None:
        super().__init__(position, rotation, trajectory, move_step)

        self.color = color
        self.halfExtents = halfExtents

    def build(self) -> int:
        self.object_id = pyb_u.create_box(position=self.position_orig,
                                          orientation=self.orientation_orig,
                                          mass=0,
                                          halfExtents=self.halfExtents,
                                          color=self.color)
        return self.object_id

class Cylinder(Obstacle):

    def __init__(self, position: Union[list, np.ndarray], rotation: Union[list, np.ndarray], trajectory: list, move_step: float, radius: float, height: float, color: list=[0.65,0.65,0.65,1]) -> None:
        super().__init__(position, rotation, trajectory, move_step)

        self.radius = radius
        self.color = color
        self.height = height

    def build(self) -> int:
        self.object_id = pyb_u.create_cylinder(position=self.position_orig,
                                               orientation=self.orientation_orig,
                                               mass=0,
                                               radius=self.radius,
                                               height=self.height,
                                               color=self.color)
        return self.object_id