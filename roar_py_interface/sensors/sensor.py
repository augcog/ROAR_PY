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
    DICT = 1,
    JSON = 2,
    YAML = 4,
    TOML = 8,
    MSGPACK = 16,
    INTERNAL_COMPRESSED = 64,

class RoarPyRemoteSupportedSensorData:
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
    def from_data(cls: type["RoarPyRemoteSupportedSensorData"], data: bytes, scheme: RoarPyRemoteSupportedSensorSerializationScheme) -> "RoarPyRemoteSupportedSensorData":
        # Not compressed data types
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

_ObsT = typing.TypeVar("_ObsT")
class RoarPySensor(typing.Generic[_ObsT]):
    def __init__(
        self, 
        name: str,
        control_timestep: float,
    ):
        self.name = name
        self.control_timestep = control_timestep

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

def custom_roar_py_sensor_wrapper(
    sensor: RoarPySensor,
    gym_observation_spec_override: typing.Optional[gym.Space],
    close_lambda: typing.Optional[typing.Callable[[RoarPySensor], None]],
    receive_observation_lambda: typing.Optional[typing.Callable[[RoarPySensor], typing.Any]],
    convert_obs_to_gym_obs_lambda: typing.Optional[typing.Callable[[RoarPySensor, typing.Any], typing.Any]],
):
    if gym_observation_spec_override is not None:
        sensor.get_gym_observation_spec = lambda: gym_observation_spec_override
    
    if close_lambda is not None:
        sensor.closed = False
        def custom_close():
            close_lambda(sensor)
            sensor.closed = True
        sensor.close = custom_close
        sensor.is_closed = lambda: sensor.closed

    if receive_observation_lambda is not None:
        sensor.last_obs = None
        def custom_receive_obs():
            sensor.last_obs = receive_observation_lambda(sensor)
            return sensor.last_obs
        sensor.receive_observation = custom_receive_obs
        sensor.get_last_observation = lambda: sensor.last_obs

    if convert_obs_to_gym_obs_lambda is not None:
        sensor.convert_obs_to_gym_obs = lambda obs: convert_obs_to_gym_obs_lambda(sensor, obs)