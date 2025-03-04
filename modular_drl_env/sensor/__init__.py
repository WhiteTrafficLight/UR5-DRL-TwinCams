from .sensor import Sensor
from .sensor_implementations import *

class SensorRegistry:
    _sensor_classes = {}

    @classmethod
    def get(cls, sensor_type:str) -> Sensor:
        try:
            return cls._sensor_classes[sensor_type]
        except KeyError:
            raise ValueError(f"unknown sensor type : {sensor_type}")

    @classmethod
    def register(cls, sensor_type:str):
        def inner_wrapper(wrapped_class):
            cls._sensor_classes[sensor_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

SensorRegistry.register('PositionRotation')(PositionRotationSensor)
SensorRegistry.register('Joints')(JointsSensor)
SensorRegistry.register('Obstacle')(ObstacleSensor)
SensorRegistry.register('ObstacleAbsolute')(ObstacleAbsoluteSensor)
SensorRegistry.register('RobotSkeleton')(RobotSkeletonSensor)
SensorRegistry.register('LidarSensorGeneric')(LidarSensorGeneric)
SensorRegistry.register('LidarSensorUR5')(LidarSensorUR5)
SensorRegistry.register('LidarSensorKR16')(LidarSensorKR16)
SensorRegistry.register('OnBodyUR5')(OnBodyCameraUR5)
SensorRegistry.register('Floating')(StaticFloatingCamera)
SensorRegistry.register('FloatingFollowEffector')(StaticFloatingCameraFollowEffector)
SensorRegistry.register('BuddyRobotCamera')(BuddyRobotCamera)
SensorRegistry.register('Time')(TimeSensor)