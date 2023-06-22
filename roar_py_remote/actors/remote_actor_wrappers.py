from roar_py_interface.base import RoarPySensor
from roar_py_interface.wrappers import RoarPyWrapper
from .remote_actors import RoarPyRemoteSharedActor, RoarPyRemoteActorObsInfo, RoarPyRemoteActorObsInfoRequest
from roar_py_interface import RoarPyActor
import typing
from ..base import RoarPyObjectWithRemoteMessage, register_object_with_remote_message
from ..sensors import RoarPyRemoteSharedSensor, RoarPyRemoteSharedSensorWrapper, RoarPyRemoteServerSensorWrapper
import gymnasium as gym
import pickle
import zlib

class RoarPyRemoteSharedActorWrapper(
    RoarPyWrapper, RoarPyRemoteSharedActor
):
    def __init__(self, wrapped_object: RoarPyActor):
        RoarPyWrapper.__init__(self, wrapped_object, "RoarPyRemoteSharedActorWrapper")
        RoarPyRemoteSharedActor.__init__(self, wrapped_object.name, wrapped_object.control_timestep, wrapped_object.force_real_control_timestep)
    
    @property
    def control_timestep(self) -> float:
        return self._wrapped_object.control_timestep
    
    @property
    def force_real_control_timestep(self) -> bool:
        return self._wrapped_object.force_real_control_timestep

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        for sensor in self._wrapped_object.get_sensors():
            if not isinstance(sensor, RoarPyRemoteSharedSensor):
                yield RoarPyRemoteSharedSensorWrapper(sensor)
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

@register_object_with_remote_message(RoarPyRemoteActorObsInfoRequest, RoarPyRemoteActorObsInfo)
class RoarPyRemoteServerActorWrapper(RoarPyWrapper, RoarPyActor, RoarPyObjectWithRemoteMessage[RoarPyRemoteActorObsInfoRequest, RoarPyRemoteActorObsInfo]):
    def __init__(self, wrapped_object: RoarPyActor):
        RoarPyWrapper.__init__(self, wrapped_object, "RoarPyRemoteServerActorWrapper")
        RoarPyObjectWithRemoteMessage.__init__(self)
        RoarPyActor.__init__(self, wrapped_object.name, wrapped_object.control_timestep, wrapped_object.force_real_control_timestep)

        self.sensors_map : typing.Dict[int, RoarPyRemoteServerSensorWrapper] = {}
        self._last_sensor_id = 0
        self._refresh_sensor_list()
    
    @property
    def control_timestep(self) -> float:
        return self._wrapped_object.control_timestep
    
    @property
    def force_real_control_timestep(self) -> bool:
        return self._wrapped_object.force_real_control_timestep

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        return self._wrapped_object.get_sensors()

    def get_action_spec(self) -> gym.Space:
        return self._wrapped_object.get_action_spec()
    
    async def _apply_action(self, action: typing.Any) -> bool:
        return await self._wrapped_object._apply_action(action)

    def close(self):
        return self._wrapped_object.close()

    def is_closed(self) -> bool:
        return self._wrapped_object.is_closed()

    def _refresh_sensor_list(self) -> None:
        new_map = {}
        actor_sensors = self._wrapped_object.get_sensors()
        for sensor in actor_sensors:
            found_in_map = False
            for oid, osensor in self.sensors_map.items():
                if sensor is osensor:
                    new_map[oid] = osensor
                    found_in_map = True
                    break
            if not found_in_map:
                new_map[self._last_sensor_id] = RoarPyRemoteServerSensorWrapper(sensor)
                self._last_sensor_id += 1
        self.sensors_map = new_map
    
    def _depack_info(self, data: RoarPyRemoteActorObsInfoRequest) -> bool:
        if data.action is not None:
            action_spec = self._wrapped_object.get_action_spec()
            try:
                real_action = gym.spaces.unflatten(action_spec, data.action)
            except:
                return False
            
            if action_spec.contains(real_action):
                self._wrapped_object.apply_action(real_action)
            else:
                return False
        
        if data.close:
            self._wrapped_object.close()
        
        self._refresh_sensor_list()
        for idx, sensor_req in data.sensors_request.items():
            if idx in self.sensors_map:
                self.sensors_map[idx]._depack_info(sensor_req)
        
        return True

    def _pack_info(self) -> RoarPyRemoteActorObsInfo:
        self._refresh_sensor_list()
        print("Packing actor info, ", self.sensors_map)
        sensor_map = dict([(idx, sensor._pack_info()) for idx, sensor in self.sensors_map.items()])
        return RoarPyRemoteActorObsInfo.from_actor(self._wrapped_object, sensor_map)