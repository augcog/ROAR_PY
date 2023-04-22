from .sensor import RoarPySensor
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

@serde
@dataclass
class RoarPyLiDARSensorData:
    # Received list of 4D points in shape (N, (X, Y, Z, I))
    # where N is the # of points,
    # X, Y, Z are the x, y, z coordinates respectively
    # I is the intensity of each array
    # Each point consists of [x, y, z] coordinates
    # plus the intensity computed for that point
    lidar_points_data: np.ndarray

class RoarPyLiDARSensor(RoarPySensor[RoarPyLiDARSensorData]):
    def __init__(
        self,
        name: str,
        control_timestep: float,
    ):
        super().__init__(name, control_timestep)

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, 4),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self, obs: RoarPyLiDARSensorData):
        return obs.lidar_points_data
