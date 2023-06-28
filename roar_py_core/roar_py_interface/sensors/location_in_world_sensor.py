from ..base import RoarPySensor,RoarPyRemoteSupportedSensorData
from ..base.sensor import remote_support_sensor_data_register
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyLocationInWorldSensorData(RoarPyRemoteSupportedSensorData):
    # Distance from origin to spot on X axis in meters
    x: float
    # Distance from origin to spot on Y axis in meters
    y: float
    # Distance from origin to spot on Z axis in meters
    z: float

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self):
        return np.array([
            self.x,
            self.y,
            self.z
        ])
    
class RoarPyLocationInWorldSensor(RoarPySensor[RoarPyLocationInWorldSensorData]):
    sensordata_type = RoarPyLocationInWorldSensorData
    def __init__(
        self, 
        name: str,
        control_timestep: float,
    ):
        super().__init__(name, control_timestep)

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )
    
    def convert_obs_to_gym_obs(self, obs: RoarPyLocationInWorldSensorData):
        return obs.convert_obs_to_gym_obs()
