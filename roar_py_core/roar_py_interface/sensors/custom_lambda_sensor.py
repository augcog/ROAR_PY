from ..base import RoarPySensor, RoarPyRemoteSupportedSensorData, RoarPyRemoteSupportedSensorSerializationScheme
from ..base.sensor import remote_support_sensor_data_register
from serde import serde
from serde.msgpack import to_msgpack, from_msgpack
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from typing import Dict, Any
import pickle
import zlib

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyCustomLambdaSensorData(RoarPyRemoteSupportedSensorData):
    flattened: np.ndarray
    space: gym.Space

    @property
    def unflattened(self):
        return gym.spaces.unflatten(self.space, self.flattened)

    def get_gym_observation_spec(self) -> gym.Space:
        return self.space

    def convert_obs_to_gym_obs(self):
        return gym.spaces.unflatten(self.space, self.flattened)
    
    def to_data(self, scheme: RoarPyRemoteSupportedSensorSerializationScheme) -> bytes:
        to_serialize = {
            "flattened": self.flattened,
            "space": zlib.compress(pickle.dumps(self.space))
        }
        return to_msgpack(to_serialize)

    @staticmethod
    def from_data_custom(data : bytes, scheme : RoarPyRemoteSupportedSensorSerializationScheme):
        dat_dict = from_msgpack(Dict[str, Any], data)
        assert "flattened" in dat_dict
        assert "space" in dat_dict
        return RoarPyCustomLambdaSensorData(
            flattened=np.asarray(dat_dict["flattened"]),
            space=pickle.loads(zlib.decompress(dat_dict["space"]))
        )

class RoarPyCustomLambdaSensor(RoarPySensor[RoarPyCustomLambdaSensorData], RoarPyRemoteSupportedSensorData):
    sensordata_type = RoarPyCustomLambdaSensorData
    def __init__(self, name: str, control_timestep: float, data_space : gym.Space):
        super().__init__(name, control_timestep)
        self.data_space = data_space
    
    def get_gym_observation_spec(self) -> gym.Space:
        return self.data_space

    def convert_obs_to_gym_obs(self, obs: RoarPyCustomLambdaSensorData):
        return obs.convert_obs_to_gym_obs()