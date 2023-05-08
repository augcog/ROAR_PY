from .sensor import RoarPySensor,RoarPyRemoteSupportedSensorData
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

@serde
@dataclass
class RoarPyGNSSSensorData(RoarPyRemoteSupportedSensorData):
    # height(m) above ground level
    altitude: float

    # degrees north(+)/south(-) on the map
    latitude: float

    # degrees east(+)/west(-) on the map from prime meridian
    longitude: float

class RoarPyGNSSSensor(RoarPySensor[RoarPyGNSSSensorData]):
    def __init__(
        self, 
        name: str,
        control_timestep: float,
    ):
        super().__init__(name, control_timestep)

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =np.array([-np.inf,-90,-180]),
            high=np.array([np.inf,90,180]),
            shape=(3,),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self, obs: RoarPyGNSSSensorData):
        return np.array([
            obs.altitude,
            obs.latitude,
            obs.longitude
        ])
