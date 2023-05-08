from roar_py_interface import RoarPyLiDARSensor, RoarPyLiDARSensorData, roar_py_thread_sync
import typing
import gymnasium as gym
import carla
import asyncio
import numpy as np
from PIL import Image
from ..base import RoarPyCarlaBase

"""
Lidar sensor data transform
https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/lidar_to_camera.py#L161
We can of course also iterate through the point cloud since it implements __iter__ and transform each point [carla.LidarDetection] individually
but this is much slower.
"""
def __convert_carla_lidar_raw_to_roar_py(carla_lidar_dat : carla.LidarMeasurement) -> RoarPyLiDARSensorData:
    p_cloud_size = len(carla_lidar_dat)
    p_cloud = np.copy(np.frombuffer(carla_lidar_dat.raw_data, dtype=np.dtype('f4')))
    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
    return RoarPyLiDARSensorData(
        carla_lidar_dat.channels,
        carla_lidar_dat.horizontal_angle,
        p_cloud
    )

class RoarPyCarlaLiDARSensor(RoarPyLiDARSensor, RoarPyCarlaBase):
    def __init__(
        self, 
        carla_instance: "RoarPyCarlaInstance",
        sensor: carla.Sensor,
        target_data_type: typing.Optional[typing.Type[RoarPyLiDARSensorData]] = None,
        name: str = "carla_lidar_sensor",
    ):
        assert sensor.type_id == "sensor.lidar.ray_cast", "Unsupported blueprint_id: {} for carla collision sensor support".format(sensor.type_id)
        RoarPyLiDARSensor.__init__(self, name, control_timestep = 0.0)
        RoarPyCarlaBase.__init__(self, carla_instance, sensor)
        self.received_data : typing.Optional[RoarPyLiDARSensorData] = None
        sensor.listen(
            self.listen_callback
        )


    @property
    def control_timestep(self) -> float:
        return self._base_actor.sensor_tick
    
    @control_timestep.setter
    @roar_py_thread_sync
    def control_timestep(self, control_timestep: float) -> None:
        self._base_actor.sensor_tick = control_timestep

    @property
    def num_lasers(self) -> int:
        return self._base_actor.channels
    
    @num_lasers.setter
    @roar_py_thread_sync
    def num_lasers(self, channels: int) -> None:
        self._base_actor.channels = channels
    
    # In meters
    @property
    def max_distance(self) -> float:
        return self._base_actor.range
    
    @max_distance.setter
    @roar_py_thread_sync
    def max_distance(self, range: float) -> None:
        self._base_actor.range = range
    
    # In meters
    @property
    def min_distance(self) -> float:
        return 0.0
    
    @property
    def points_per_second(self) -> int:
        return self._base_actor.points_per_second
    
    @points_per_second.setter
    @roar_py_thread_sync
    def points_per_second(self, points_per_second: int) -> None:
        self._base_actor.points_per_second = points_per_second
    
    @property
    def rotation_frequency(self) -> float:
        return self._base_actor.rotation_frequency
    
    @rotation_frequency.setter
    @roar_py_thread_sync
    def rotation_frequency(self, rotation_frequency: float) -> None:
        self._base_actor.rotation_frequency = rotation_frequency

    @property
    def upper_fov(self) -> float:
        return self._base_actor.upper_fov
    
    @upper_fov.setter
    @roar_py_thread_sync
    def upper_fov(self, upper_fov: float) -> None:  
        self._base_actor.upper_fov = upper_fov  
    
    @property
    def lower_fov(self) -> float:
        return self._base_actor.lower_fov
    
    @lower_fov.setter
    @roar_py_thread_sync
    def lower_fov(self, lower_fov: float) -> None:
        self._base_actor.lower_fov = lower_fov
    
    @property
    def horizontal_fov(self) -> int:
        return self._base_actor.horizontal_fov
    
    @horizontal_fov.setter
    @roar_py_thread_sync
    def horizontal_fov(self, horizontal_fov: float) -> None:
        self._base_actor.horizontal_fov = horizontal_fov

    async def receive_observation(self) -> RoarPyLiDARSensorData:
        while self.received_data is None:
            await asyncio.sleep(0.001)
        return self.received_data
    
    def listen_carla_data(self, carla_data: carla.LidarMeasurement) -> None:
        self.received_data = __convert_carla_lidar_raw_to_roar_py(carla_data)

    def get_last_observation(self) -> typing.Optional[RoarPyLiDARSensorData]:
        return self.received_data
    
    @roar_py_thread_sync
    def close(self):
        if self._base_actor is not None and self._base_actor.is_listening:
            self._base_actor.stop()
        RoarPyCarlaBase.close(self)
    
    @roar_py_thread_sync
    def is_closed(self) -> bool:
        return self.sensor is None or not self.sensor.is_listening
