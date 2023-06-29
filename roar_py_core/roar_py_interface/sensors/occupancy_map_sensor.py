from roar_py_interface.worlds.occupancy_map import RoarPyOccupancyMapProducer
from ..base.sensor import RoarPySensor, RoarPyRemoteSupportedSensorData, remote_support_sensor_data_register, RoarPyRemoteSupportedSensorSerializationScheme
from .location_in_world_sensor import RoarPyLocationInWorldSensor
from .rotation_sensor import RoarPyRollPitchYawSensor
from ..worlds.waypoint import RoarPyWaypoint
from ..worlds.occupancy_map import RoarPyOccupancyMapProducer
from serde import serde
from dataclasses import dataclass
from PIL import Image
import numpy as np
import gymnasium as gym
import io
from typing import Optional, Tuple, Any, Callable, Awaitable

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyOccupancyMapSensorData(RoarPyRemoteSupportedSensorData):
    occupancy_map: Image.Image
    
    def convert_obs_to_gym_obs(self):
        return np.asarray(self.occupancy_map, dtype=np.uint8).reshape((*self.occupancy_map.size, 1))

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        return gym.spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
    
    def get_gym_observation_spec(self) -> gym.Space:
        return self.__class__.gym_observation_space(*self.occupancy_map.size)

    @staticmethod
    def from_image(image: Image.Image):
        return __class__(
            image.convert("L")
        )

    def to_data(self, scheme: RoarPyRemoteSupportedSensorSerializationScheme) -> bytes:
        saved_image = io.BytesIO()
        self.occupancy_map.save(saved_image, format="JPEG")
        return saved_image.getvalue()

    @staticmethod
    def from_data_custom(data : bytes, scheme : RoarPyRemoteSupportedSensorSerializationScheme):
        image_bytes = io.BytesIO(data)
        image_bytes.seek(0)
        img = Image.open(image_bytes)
        return __class__.from_image(img)

class RoarPyOccupancyMapSensor(RoarPySensor[RoarPyOccupancyMapSensorData]):
    sensordata_type = RoarPyOccupancyMapSensorData
    def __init__(self, name: str = "occupancy_map_sensor"):
        super().__init__(name, 0.0)

class RoarPyOccupancyMapSensorImpl(RoarPyOccupancyMapSensor):
    def __init__(self, producer : RoarPyOccupancyMapProducer, location_sensor : RoarPyLocationInWorldSensor, rotation_sensor : RoarPyRollPitchYawSensor, name: str = "occupancy_map_sensor"):
        super().__init__(name)
        self._closed = False
        self.producer = producer
        self.location_sensor = location_sensor
        self.rotation_sensor = rotation_sensor
        self._last_data : Optional[RoarPyOccupancyMapSensorData] = None
    
    async def get_2d_location(self) -> np.ndarray:
        location = await self.location_sensor.receive_observation()
        return np.array([location.x, location.y])
    
    async def get_rotation_yaw(self) -> float:
        rotation = await self.rotation_sensor.receive_observation()
        return rotation.roll_pitch_yaw[2]
    
    def get_gym_observation_spec(self) -> gym.Space:
        return RoarPyOccupancyMapSensorData.gym_observation_space(self.producer.width, self.producer.height)
    
    async def receive_observation(self) -> RoarPyOccupancyMapSensorData:
        location = await self.get_2d_location()
        yaw = await self.get_rotation_yaw()
        occupancy_map = self.producer.plot_occupancy_map(location, yaw)
        self._last_data = RoarPyOccupancyMapSensorData.from_image(occupancy_map)
        return self._last_data
    
    def get_last_observation(self) -> Optional[RoarPyOccupancyMapSensorData]:
        return self._last_data

    def convert_obs_to_gym_obs(self, obs: RoarPyOccupancyMapSensorData):
        return obs.convert_obs_to_gym_obs()
    
    def close(self):
        self._closed = True
    
    def is_closed(self) -> bool:
        return self._closed