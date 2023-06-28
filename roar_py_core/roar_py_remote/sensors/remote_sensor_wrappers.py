from .remote_sensors import RoarPyRemoteSharedSensor, RoarPyRemoteSensorObsInfo, RoarPyRemoteSensorObsInfoRequest
from ..base import RoarPyObjectWithRemoteMessage, register_object_with_remote_message
from roar_py_interface.wrappers import RoarPySensorWrapper
from roar_py_interface import RoarPySensor, RoarPyCollisionSensorData, RoarPyRemoteSupportedSensorData
import gymnasium as gym
import typing

_ObsT = typing.TypeVar("_ObsT")
class RoarPyRemoteSharedSensorWrapper(typing.Generic[_ObsT], RoarPyRemoteSharedSensor[_ObsT], RoarPySensorWrapper[_ObsT]):
    def __init__(self, wrapped_object : RoarPySensor[_ObsT]):
        RoarPySensorWrapper.__init__(self, wrapped_object, "RoarPyRemoteSharedSensorWrapper")
        RoarPyRemoteSharedSensor.__init__(self, wrapped_object.name, wrapped_object.control_timestep)

    @property
    def control_timestep(self) -> float:
        return self._wrapped_object.control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value: float):
        raise AttributeError("Cannot set control_timestep on a remote sensor")
    
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

_ObsTServer = typing.TypeVar("_ObsTServer", bound=RoarPyRemoteSupportedSensorData)

@register_object_with_remote_message(RoarPyRemoteSensorObsInfoRequest, RoarPyRemoteSensorObsInfo)
class RoarPyRemoteServerSensorWrapper(typing.Generic[_ObsTServer], RoarPySensorWrapper[_ObsTServer], RoarPyObjectWithRemoteMessage[RoarPyRemoteSensorObsInfoRequest,RoarPyRemoteSensorObsInfo]):
    def __init__(self, sensor: RoarPySensor[_ObsTServer]):
        RoarPySensorWrapper.__init__(self, sensor, "RoarPyRemoteServerSensor")
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