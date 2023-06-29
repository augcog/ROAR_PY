from roar_py_interface.base import RoarPySensor
from .remote_actors import RoarPyRemoteSharedActor, RoarPyRemoteActorObsInfo, RoarPyRemoteActorObsInfoRequest
from roar_py_interface import RoarPyActor, RoarPyActorWrapper
import typing
from ..base import RoarPyObjectWithRemoteMessage, register_object_with_remote_message
from ..sensors import RoarPyRemoteSharedSensor, RoarPyRemoteSharedSensorWrapper, RoarPyRemoteServerSensorWrapper
import gymnasium as gym
import asyncio


class RoarPyRemoteSharedActorWrapper(
    RoarPyActorWrapper, RoarPyRemoteSharedActor
):
    def __init__(self, wrapped_object: RoarPyActor):
        RoarPyActorWrapper.__init__(self, wrapped_object, "RoarPyRemoteSharedActorWrapper")
        RoarPyRemoteSharedActor.__init__(self, wrapped_object.name, wrapped_object.control_timestep, wrapped_object.force_real_control_timestep)

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        for sensor in self._wrapped_object.get_sensors():
            if not isinstance(sensor, RoarPyRemoteSharedSensor):
                yield RoarPyRemoteSharedSensorWrapper(sensor)
            else:
                yield sensor

@register_object_with_remote_message(RoarPyRemoteActorObsInfoRequest, RoarPyRemoteActorObsInfo)
class RoarPyRemoteServerActorWrapper(RoarPyActorWrapper, RoarPyObjectWithRemoteMessage[RoarPyRemoteActorObsInfoRequest, RoarPyRemoteActorObsInfo]):
    def __init__(self, wrapped_object: RoarPyActor):
        RoarPyActorWrapper.__init__(self, wrapped_object, "RoarPyRemoteServerActorWrapper")
        RoarPyObjectWithRemoteMessage.__init__(self)

        self.sensors_map : typing.Dict[int, RoarPyRemoteServerSensorWrapper] = {}
        self._last_sensor_id = 0
        self._need_action_space_spec = True
        self._next_action = None
        self._refresh_sensor_list()

    async def receive_observation(self):
        if self._next_action is not None:
            await self.apply_action(self._next_action)
            self._next_action = None
        
        return await self._wrapped_object.receive_observation()

    def _refresh_sensor_list(self) -> None:
        new_map = {}
        actor_sensors = self._wrapped_object.get_sensors()
        for sensor in actor_sensors:
            found_in_map = False
            for oid, osensor in self.sensors_map.items():
                if sensor is osensor._wrapped_object:
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
                self._next_action = None
                return False
            
            if action_spec.contains(real_action):
                # TODO: Fix this
                self._next_action = real_action
            else:
                self._next_action = None
                return False
        else:
            self._next_action = None
        
        if data.close:
            self._wrapped_object.close()
        
        self._need_action_space_spec = data.need_action_space_spec
        
        self._refresh_sensor_list()

        for idx, sensor_req in data.sensors_request.items():
            if idx in self.sensors_map and sensor_req is not None:
                self.sensors_map[idx]._depack_info(sensor_req)

        return True

    def _pack_info(self) -> RoarPyRemoteActorObsInfo:
        self._refresh_sensor_list()
        sensor_map = dict([(idx, sensor._pack_info()) for idx, sensor in self.sensors_map.items()])
        return RoarPyRemoteActorObsInfo.from_actor(self._wrapped_object, sensor_map, self._need_action_space_spec)

    async def _tick_remote(self):
        await self.receive_observation()