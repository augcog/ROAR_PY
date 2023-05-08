from roar_py_interface.sensors.sensor import RoarPySensor
from roar_py_interface.base import RoarPyWrapper
from .remote_actors import RoarPyRemoteSharedActor
from roar_py_interface import RoarPyActor
import typing
from ..sensors import RoarPyRemoteSharedSensor, RoarPyRemoteSharedSensorWrapper
import gymnasium as gym

class RoarPyRemoteSharedActorWrapper(
    RoarPyWrapper, RoarPyRemoteSharedActor
):
    def __init__(self, wrapped_object: RoarPyActor, wrapper_name: str):
        RoarPyWrapper.__init__(self, wrapped_object, wrapper_name)
        RoarPyRemoteSharedActor.__init__(self, wrapped_object, wrapper_name)
    
    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        for sensor in self._wrapped_object.get_sensors():
            if not isinstance(sensor, RoarPyRemoteSharedSensor):
                yield RoarPyRemoteSharedSensorWrapper(sensor, self._wrapper_name + ".get_sensors()")
            else:
                yield sensor

    def get_action_spec(self) -> gym.Space:
        return self._wrapped_object.get_action_spec()

    def is_closed(self) -> bool:
        return self._wrapped_object.is_closed()
    
    def get_gym_observation_spec(self) -> gym.Space:
        return self._wrapped_object.get_gym_observation_spec()

    def get_last_observation(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        return self._wrapped_object.get_last_observation()
    
    def get_last_gym_observation(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        return self._wrapped_object.get_last_gym_observation()

    def convert_obs_to_gym_obs(self, observation: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        return self._wrapped_object.convert_obs_to_gym_obs(observation)
