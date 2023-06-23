from .remote_sensors import RoarPyRemoteSharedSensor, RoarPyRemoteSensorObsInfo, RoarPyRemoteSensorObsInfoRequest
from ..base import RoarPyObjectWithRemoteMessage, register_object_with_remote_message
from roar_py_interface.wrappers import RoarPyWrapper
from roar_py_interface import RoarPySensor, RoarPyCollisionSensorData, RoarPyRemoteSupportedSensorData
import gymnasium as gym
import typing

_ObsT = typing.TypeVar("_ObsT")
class RoarPyRemoteSharedSensorWrapper(typing.Generic[_ObsT], RoarPyRemoteSharedSensor[_ObsT], RoarPyWrapper[RoarPySensor[_ObsT]]):
    def __init__(self, wrapped_object : RoarPySensor[_ObsT]):
        RoarPyWrapper.__init__(self, wrapped_object, "RoarPyRemoteSharedSensor")
        RoarPyRemoteSharedSensor.__init__(self, wrapped_object.name, wrapped_object.control_timestep)

    @property
    def control_timestep(self) -> float:
        return self._wrapped_object.control_timestep
    
    @property
    def name(self) -> str:
        return self._wrapped_object.name
    
    def get_gym_observation_spec(self) -> gym.Space:
        return self._wrapped_object.get_gym_observation_spec()
    
    def get_last_observation(self) -> typing.Optional[_ObsT]:
        last_obs = self._wrapped_object.get_last_observation()
        if last_obs is not None and isinstance(last_obs, RoarPyCollisionSensorData):
            # Mask out collided objects
            last_obs = RoarPyCollisionSensorData(
                actor=None,
                other_actor=None,
                impulse_normal=last_obs.impulse_normal,
            )
        return last_obs

    def convert_obs_to_gym_obs(self, obs: _ObsT):
        return self._wrapped_object.convert_obs_to_gym_obs(obs)
     
    def is_closed(self) -> bool:
        return self._wrapped_object.is_closed()

    def get_last_gym_observation(self) -> typing.Optional[typing.Any]:
        return self._wrapped_object.get_last_gym_observation()

_ObsTServer = typing.TypeVar("_ObsTServer", bound=RoarPyRemoteSupportedSensorData)

@register_object_with_remote_message(RoarPyRemoteSensorObsInfoRequest, RoarPyRemoteSensorObsInfo)
class RoarPyRemoteServerSensorWrapper(typing.Generic[_ObsTServer], RoarPySensor[_ObsTServer], RoarPyWrapper[RoarPyRemoteSensorObsInfo], RoarPyObjectWithRemoteMessage[RoarPyRemoteSensorObsInfoRequest,RoarPyRemoteSensorObsInfo]):
    def __init__(self, sensor: RoarPySensor[_ObsTServer]):
        RoarPyWrapper.__init__(self, sensor, "RoarPyRemoteServerSensor")
        RoarPySensor.__init__(self, sensor.name, sensor.control_timestep)
        RoarPyObjectWithRemoteMessage.__init__(self)
        self._pack_obs_spec = True
    
    def _depack_info(self, data: RoarPyRemoteSensorObsInfoRequest) -> bool:
        if data.close:
            self.close()
        self._pack_obs_spec = data.need_obs_spec
        return True
    
    def _pack_info(self) -> RoarPyRemoteSensorObsInfo:
        return RoarPyRemoteSensorObsInfo.from_sensor(self, self._pack_obs_spec)
    
    async def _tick_remote(self):
        await self.receive_observation()

    @property
    def name(self) -> str:
        return self._wrapped_object.name
    
    @name.setter
    def name(self, value: str) -> None:
        self._wrapped_object.name = value

    @property
    def control_timestep(self) -> float:
        return super().control_timestep
    
    def get_gym_observation_spec(self) -> gym.Space:
        return self._wrapped_object.get_gym_observation_spec()
    
    async def receive_observation(self) -> _ObsT:
        return await self._wrapped_object.receive_observation()
    
    def get_last_observation(self) -> typing.Optional[_ObsT]:
        return self._wrapped_object.get_last_observation()

    def convert_obs_to_gym_obs(self, obs: _ObsT):
        return self._wrapped_object.convert_obs_to_gym_obs(obs)
    
    def close(self):
        self._wrapped_object.close()
    
    def is_closed(self) -> bool:
        return self._wrapped_object.is_closed()