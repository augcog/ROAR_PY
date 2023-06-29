from ..base import RoarPySensor, RoarPyRemoteSupportedSensorData
from ..base.sensor import remote_support_sensor_data_register
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyAccelerometerSensorData(RoarPyRemoteSupportedSensorData):
    # acceleration (x,y,z local axis) in m/s^2
    acceleration: np.ndarray #np.NDArray[np.float32]

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self):
        return self.acceleration

class RoarPyAccelerometerSensor(RoarPySensor[RoarPyAccelerometerSensorData], RoarPyRemoteSupportedSensorData):
    sensordata_type = RoarPyAccelerometerSensorData
    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self, obs: RoarPyAccelerometerSensorData):
        return obs.convert_obs_to_gym_obs()