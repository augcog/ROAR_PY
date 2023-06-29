from ..base import RoarPySensor,RoarPyRemoteSupportedSensorData
from ..base.sensor import remote_support_sensor_data_register
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyGNSSSensorData(RoarPyRemoteSupportedSensorData):
    # height(m) above ground level
    altitude: float

    # degrees north(+)/south(-) on the map
    latitude: float

    # degrees east(+)/west(-) on the map from prime meridian
    longitude: float

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =np.array([-np.inf,-90,-180]),
            high=np.array([np.inf,90,180]),
            shape=(3,),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self):
        return np.array([
            self.altitude,
            self.latitude,
            self.longitude
        ])

class RoarPyGNSSSensor(RoarPySensor[RoarPyGNSSSensorData]):
    sensordata_type = RoarPyGNSSSensorData
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
        return obs.convert_obs_to_gym_obs()