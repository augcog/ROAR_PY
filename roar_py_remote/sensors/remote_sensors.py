from roar_py_interface import RoarPySensor, RoarPyRemoteSupportedSensorData, RoarPyRemoteSupportedSensorSerializationScheme
from typing import Any, TypeVar, Generic, Optional
import gymnasium as gym
from serde import serde
from dataclasses import dataclass
import pickle
import zlib
import asyncio

_ObsT = TypeVar("_ObsT")
class RoarPyRemoteSharedSensor(RoarPySensor[_ObsT], Generic[_ObsT]):
    async def receive_observation(self) -> Any:
        raise PermissionError("This sensor is shared, you cannot receive observation from it")
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        raise PermissionError("Cannot set attribute on a shared actor")

    def __delattr__(self, __name: str) -> None:
        raise PermissionError("Cannot delete attribute on a shared actor")

    def close(self):
        pass

@serde
@dataclass
class RoarPyRemoteSensorInfo:
    name: Optional[str]
    control_timestep: float
    last_data: bytes
    last_data_type: str
    obs_spec: Optional[bytes]
    is_closed: bool

@serde
@dataclass
class RoarPyRemoteSensorInfoRequest:
    new_obs : bool
    close : bool

_ObsTClient = TypeVar("_ObsTClient", bound=RoarPyRemoteSupportedSensorData)
class RoarPyRemoteClientSensor(RoarPySensor[_ObsTClient], Generic[_ObsTClient]):
    def __init__(
        self,
        start_info: RoarPyRemoteSensorInfo,
    ):
        RoarPySensor.__init__(self, name = start_info.name if start_info.name is not None else "RoarPyRemoteClientSensor", control_timestep = start_info.control_timestep)
        self._closed = False
        self._last_data = None
        self._new_data = None
        self._obs_spec = None
        self.new_request : RoarPyRemoteSensorInfoRequest = RoarPyRemoteSensorInfoRequest(
            False,
            False
        )
        self._depack_info(start_info)
    
    def _depack_info(self, data: RoarPyRemoteSensorInfo) -> bool:
        assert data.last_data_type in RoarPyRemoteSupportedSensorData._supported_data_types, f"Unsupported data type {data.last_data_type}"
        self._control_timestep = data.control_timestep
        
        last_data_type_real = RoarPyRemoteSupportedSensorData._supported_data_types[data.last_data_type]
        try:
            self._new_data = last_data_type_real.from_data(
                data.last_data,
                RoarPyRemoteSupportedSensorSerializationScheme.MSGPACK
            )
        except:
            pass
        if data.obs_spec is not None:
            try:
                self._obs_spec = pickle.loads(zlib.decompress(data.obs_spec))
            except:
                pass
        self._closed = data.is_closed

    def get_gym_observation_spec(self) -> gym.Space:
        if self._obs_spec is None:
            raise RuntimeError("Observation spec is not available")
        return self._obs_spec
    
    async def receive_observation(self) -> _ObsT:
        while self._new_data is None:
            await asyncio.sleep(0.001)
        self._last_data = self._new_data
        return self._last_data
    
    def get_last_observation(self) -> Optional[_ObsT]:
        return self._last_data

    def convert_obs_to_gym_obs(self, obs: _ObsT):
        assert isinstance(obs, RoarPyRemoteSupportedSensorData)
        return obs.convert_obs_to_gym_obs()
    
    def close(self):
        self.new_request.close = True
    
    def is_closed(self) -> bool:
        return self._closed