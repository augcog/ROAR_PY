from ..base import RoarPySensor, RoarPyRemoteSupportedSensorData
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

@serde
@dataclass
class RoarPyLiDARSensorData(RoarPyRemoteSupportedSensorData):
    # Number of lasers shot
    channels: int

    # Horizontal angle the LIDAR is rotated 
    # at the time of the measurement
    horizontal_angle: float
    
    # Received list of 4D points in shape (N, (X, Y, Z, I))
    # where N is the # of points,
    # X, Y, Z are the x, y, z coordinates respectively
    # I is the intensity of each array
    # Each point consists of [x, y, z] coordinates
    # plus the intensity computed for that point
    # intensity is a value between 0 and 1
    lidar_points_data: np.ndarray


class RoarPyLiDARSensor(RoarPySensor[RoarPyLiDARSensorData]):
    def __init__(
        self,
        name: str,
        control_timestep: float
    ):
        super().__init__(name, control_timestep)

    @property
    def num_lasers(self) -> int:
        raise NotImplementedError
    
    # In meters
    @property
    def max_distance(self) -> float:
        raise NotImplementedError
    
    # In meters
    @property
    def min_distance(self) -> float:
        raise NotImplementedError
    
    @property
    def points_per_second(self) -> int:
        raise NotImplementedError
    
    @property
    def rotation_frequency(self) -> float:
        raise NotImplementedError

    @property
    def upper_fov(self) -> float:
        raise NotImplementedError
    
    @property
    def lower_fov(self) -> float:
        raise NotImplementedError
    
    @property
    def horizontal_fov(self) -> int:
        raise NotImplementedError
    
    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, 4),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self, obs: RoarPyLiDARSensorData):
        return obs.lidar_points_data
