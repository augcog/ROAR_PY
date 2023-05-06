from roar_py_interface import RoarPyLiDARSensor, RoarPyLiDARSensorData
import typing
import gymnasium as gym
import carla
import asyncio
import numpy as np
from PIL import Image
from ..base import RoarPyCarlaBase

class RoarPyCarlaLiDARSensor(RoarPyLiDARSensor[RoarPyLiDARSensorData], RoarPyCarlaBase):
    def __init__(
        self, 
        sensor: carla.Sensor,
        target_data_type: typing.Optional[typing.Type[RoarPyLiDARSensorData]] = None,
        name: str = "carla_lidar_sensor",
    ):
        assert sensor.type_id == "sensor.lidar.ray_cast", "Unsupported blueprint_id: {} for carla collision sensor support".format(sensor.type_id)
        RoarPyLiDARSensor.__init__(self, name, control_timestep = 0.0)
        RoarPyCarlaBase.__init__(self, sensor)
        self.received_data : typing.Optional[RoarPyLiDARSensorData] = None
        sensor.listen(
            self.listen_callback
        )


    @property
    def control_timestep(self) -> float:
        return self._base_actor.sensor_tick
    
    @control_timestep.setter
    def control_timestep(self, control_timestep: float) -> None:
        self._base_actor.sensor_tick = control_timestep

    @property
    def num_lasers(self) -> int:
        return self._base_actor.channels
    
    @num_lasers.setter
    def num_lasers(self, channels: int) -> None:
        self._base_actor.channels = channels
    
    # In meters
    @property
    def max_distance(self) -> float:
        return self._base_actor.range
    
    @max_distance.setter
    def max_distance(self, range: float) -> None:
        self._base_actor.range = range
    
    # In meters
    @property
    def min_distance(self) -> float:
        raise NotImplementedError
    
    @min_distance.setter
    def min_distance(self, min_distance: float) -> None:
        raise NotImplementedError
    
    @property
    def points_per_second(self) -> int:
        return self._base_actor.points_per_second
    
    @points_per_second.setter
    def points_per_second(self, points_per_second: int) -> None:
        self._base_actor.points_per_second = points_per_second
    
    @property
    def rotation_frequency(self) -> float:
        return self._base_actor.rotation_frequency
    
    @rotation_frequency.setter
    def rotation_frequency(self, rotation_frequency: float) -> None:
        self._base_actor.rotation_frequency = rotation_frequency

    @property
    def upper_fov(self) -> float:
        return self._base_actor.upper_fov
    
    @upper_fov.setter
    def upper_fov(self, upper_fov: float) -> None:  
        self._base_actor.upper_fov = upper_fov  
    
    @property
    def lower_fov(self) -> float:
        return self._base_actor.lower_fov
    
    @lower_fov.setter
    def lower_fov(self, lower_fov: float) -> None:
        self._base_actor.lower_fov = lower_fov
    
    @property
    def horizontal_fov(self) -> int:
        return self._base_actor.horizontal_fov
    
    @horizontal_fov.setter
    def horizontal_fov(self, horizontal_fov: float) -> None:
        self._base_actor.horizontal_fov = horizontal_fov

    async def receive_observation(self) -> RoarPyLiDARSensorData:
        while self.received_data is None:
            await asyncio.sleep(0.001)
        return self.received_data
    
    def get_gym_observation_spec(self) -> gym.Space:
        return RoarPyLiDARSensorData.get_gym_observation_spec()
    
    def listen_carla_data(self, carla_data: carla.LidarMeasurement) -> None:
        self.received_data = RoarPyLiDARSensorData(
            carla_data.channels,
            carla_data.horizontal_angle,
            np.array(carla_data.raw_data)
        )

    def get_last_observation(self) -> typing.Optional[RoarPyLiDARSensorData]:
        return self.received_data
    
    def close(self):
        if self._base_actor is not None and self._base_actor.is_listening:
            self._base_actor.stop()
        RoarPyCarlaBase.close(self)
    
    def is_closed(self) -> bool:
        return self.sensor is None or not self.sensor.is_listening
