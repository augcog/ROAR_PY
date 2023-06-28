import typing
import asyncio
import gymnasium as gym
from enum import Enum
from dataclasses import dataclass
import serde
import serde.json
import serde.yaml
import serde.toml
import serde.msgpack
import serde.pickle
import zlib

class RoarPyRemoteSupportedSensorSerializationScheme(Enum):
    DICT = 1
    JSON = 2
    YAML = 4
    TOML = 8
    MSGPACK = 16
    INTERNAL_COMPRESSED = 64
    JSON_COMPRESSED = 2 | 64
    YAML_COMPRESSED = 4 | 64
    TOML_COMPRESSED = 8 | 64
    MSGPACK_COMPRESSED = 16 | 64


class RoarPyRemoteSupportedSensorData:
    _supported_data_types : typing.Dict[str, typing.Type["RoarPyRemoteSupportedSensorData"]] = {}
    def to_data(self, scheme : RoarPyRemoteSupportedSensorSerializationScheme) -> typing.Any:
        # Not compressed data types
        if scheme == RoarPyRemoteSupportedSensorSerializationScheme.DICT:
            ret = serde.to_dict(self)
            return ret

        # Compressed data types
        if scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.JSON.value > 0:
            ret = serde.json.to_json(self)
            ret = ret.encode("utf-8")
        elif scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.YAML.value > 0:
            ret = serde.yaml.to_yaml(self)
            ret = ret.encode("utf-8")
        elif scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.TOML.value > 0:
            ret = serde.toml.to_toml(self)
            ret = ret.encode("utf-8")
        elif scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.MSGPACK.value > 0:
            ret = serde.msgpack.to_msgpack(self)
        else:
            raise NotImplementedError()

        if scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.INTERNAL_COMPRESSED.value > 0:
            ret = zlib.compress(ret)

        return ret
    
    @classmethod
    def from_data(cls: typing.Type["RoarPyRemoteSupportedSensorData"], data: bytes, scheme: RoarPyRemoteSupportedSensorSerializationScheme) -> "RoarPyRemoteSupportedSensorData":
        # Not compressed data types
        if hasattr(cls, "from_data_custom"):
            ret = cls.from_data_custom(data, scheme)
            return ret
        
        if scheme == RoarPyRemoteSupportedSensorSerializationScheme.DICT:
            ret = serde.from_dict(cls, data)
            return ret
        
        # Compressed data types
        if scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.INTERNAL_COMPRESSED.value > 0:
            data = zlib.decompress(data)
        
        if scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.JSON.value > 0:
            ret = serde.json.from_json(cls, data.decode("utf-8"))
        elif scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.YAML.value > 0:
            ret = serde.yaml.from_yaml(cls, data.decode("utf-8"))
        elif scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.TOML.value > 0:
            ret = serde.toml.from_toml(cls, data.decode("utf-8"))
        elif scheme.value & RoarPyRemoteSupportedSensorSerializationScheme.MSGPACK.value > 0:
            ret = serde.msgpack.from_msgpack(cls, data)
        else:
            raise NotImplementedError()
        
        return ret
    
    def get_gym_observation_spec(self) -> gym.Space:
        raise NotImplementedError()

    def convert_obs_to_gym_obs(self):
        raise NotImplementedError()

def remote_support_sensor_data_register(cls): #: typing.Type["RoarPyRemoteSupportedSensorData"]):
    RoarPyRemoteSupportedSensorData._supported_data_types[cls.__name__] = cls
    return cls

_ObsT = typing.TypeVar("_ObsT")
class RoarPySensor(typing.Generic[_ObsT]):
    sensordata_type : typing.Type = _ObsT
    def __init__(
        self, 
        name: str,
        control_timestep: float,
    ):
        self.name = name
        self._control_timestep = control_timestep

    @property
    def control_timestep(self) -> float:
        return self._control_timestep

    def get_gym_observation_spec(self) -> gym.Space:
        raise NotImplementedError()
    
    async def receive_observation(self) -> _ObsT:
        raise NotImplementedError()
    
    def get_last_observation(self) -> typing.Optional[_ObsT]:
        raise NotImplementedError()

    def convert_obs_to_gym_obs(self, obs: _ObsT):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()
    
    def is_closed(self) -> bool:
        raise NotImplementedError()

    def __del__(self):
        try:
            if not self.is_closed():
                self.close()
        finally:
            pass

    def get_last_gym_observation(self) -> typing.Optional[typing.Any]:
        return self.convert_obs_to_gym_obs(self.get_last_observation())
