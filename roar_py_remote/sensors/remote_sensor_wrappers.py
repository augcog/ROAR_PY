from .remote_sensors import RoarPyRemoteSharedSensor
from roar_py_interface.wrappers import RoarPyWrapper
from roar_py_interface import RoarPySensor
import gymnasium as gym
import typing

_ObsT = typing.TypeVar("_ObsT")
class RoarPyRemoteSharedSensorWrapper(typing.Generic[_ObsT], RoarPyRemoteSharedSensor[_ObsT], RoarPyWrapper[RoarPyRemoteSharedSensor[_ObsT]]):
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
        return self._wrapped_object.get_last_observation()

    def convert_obs_to_gym_obs(self, obs: _ObsT):
        return self._wrapped_object.convert_obs_to_gym_obs(obs)
     
    def is_closed(self) -> bool:
        return self._wrapped_object.is_closed()

    def get_last_gym_observation(self) -> typing.Optional[typing.Any]:
        return self._wrapped_object.get_last_gym_observation()
