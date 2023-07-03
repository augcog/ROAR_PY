from typing import Iterable, Union, List, Dict, Any, Optional
import gymnasium as gym
from roar_py_interface.actors import RoarPyActor
from roar_py_interface.base import RoarPySensor
from roar_py_interface.base.sensor import RoarPySensor
from roar_py_interface.wrappers.wrapper_base import RoarPyWrapper
from .wrapper_base import RoarPyActorWrapper

class RoarPyActorSensorFilterWrapper(RoarPyActorWrapper):
    def __init__(
        self, 
        wrapped_object: Union[RoarPyActor,RoarPyWrapper[RoarPyActor]], 
        exclude_sensors: List[RoarPySensor] = [],
        wrapper_name: str = "RoarPyActorSensorFilterWrapper"
    ):
        super().__init__(wrapped_object, wrapper_name)
        self.exclude_sensors = exclude_sensors
    
    def get_sensors(self) -> Iterable[RoarPySensor]:
        original_list = self._wrapped_object.get_sensors()
        return filter(lambda x: x not in self.exclude_sensors, original_list)

    async def receive_observation(self) -> Dict[str, Any]:
        return await RoarPyActor.receive_observation(self)
    
    def get_last_observation(self) -> Optional[Dict[str, Any]]:
        return self._last_obs

    def get_last_gym_observation(self) -> Optional[Dict[str, Any]]:
        return self.convert_obs_to_gym_obs(
            self.get_last_observation()
        )

    def get_gym_observation_spec(self) -> gym.Space:
        return RoarPyActor.get_gym_observation_spec(self)

    def convert_obs_to_gym_obs(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return RoarPyActor.convert_obs_to_gym_obs(self, observation)