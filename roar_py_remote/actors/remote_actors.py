from typing import Any
from roar_py_interface.actors.actor import RoarPyActor, RoarPyResettableActor
from roar_py_interface.sensors import *
from roar_py_interface.base import RoarPySensor
import typing
import gymnasium as gym
import carla
import transforms3d as tr3d
import numpy as np
from serde import serde
from dataclasses import dataclass
import pickle
import zlib

from ..base import RoarPyObjectWithRemoteMessage, register_object_with_remote_message
from ..sensors.remote_sensors import RoarPyRemoteSensorObsInfo, RoarPyRemoteSensorObsInfoRequest, RoarPyRemoteClientSensor
from ..sensors.remote_sensor_wrappers import RoarPyRemoteServerSensorWrapper

class RoarPyRemoteSharedActor(RoarPyActor):
    async def apply_action(self, action: typing.Any) -> bool:
        return False # No action should be applied if this is a shared actor
    
    async def receive_observation(self) -> typing.Dict[str,typing.Any]:
        raise PermissionError("Cannot receive observation from a shared actor")

    def __setattr__(self, __name: str, __value: Any) -> None:
        raise PermissionError("Cannot set attribute on a shared actor")

    def __delattr__(self, __name: str) -> None:
        raise PermissionError("Cannot delete attribute on a shared actor")

    def close(self) -> None:
        pass

    async def reset(self) -> None:
        if isinstance(self._wrapped_object, RoarPyResettableActor):
            raise PermissionError("Cannot reset a shared actor")
        else:
            raise PermissionError("Cannot reset a non-resettable actor")

@serde
@dataclass
class RoarPyRemoteActorObsInfo:
    name: typing.Optional[str]
    control_timestep: float
    sensors_map = typing.Dict[int, RoarPyRemoteSensorObsInfo]
    is_closed: bool
    action_spec: typing.Optional[bytes]

    def get_action_spec(self) -> typing.Optional[gym.Space]:
        if self.action_spec is None:
            return None
        return pickle.loads(zlib.decompress(self.action_spec))

    @staticmethod
    def from_actor(actor: RoarPyActor, sensor_data_map : typing.Dict[int, RoarPyRemoteSensorObsInfo]) -> "RoarPyRemoteActorObsInfo":
        return RoarPyRemoteActorObsInfo(
            name=actor.name,
            control_timestep=actor.control_timestep,
            sensors_map=sensor_data_map,
            is_closed=actor.is_closed(),
            action_spec=zlib.compress(pickle.dumps(actor.get_action_spec()))
        )

@serde
@dataclass
class RoarPyRemoteActorObsInfoRequest:
    close: bool
    sensors_request: typing.Dict[int, typing.Optional[RoarPyRemoteSensorObsInfoRequest]]
    action: typing.Optional[np.ndarray]

@register_object_with_remote_message(RoarPyRemoteActorObsInfo, RoarPyRemoteActorObsInfoRequest)
class RoarPyRemoteClientActor(RoarPyActor, RoarPyObjectWithRemoteMessage[RoarPyRemoteActorObsInfo, RoarPyRemoteActorObsInfoRequest]):
    def __init__(self, start_info : RoarPyRemoteActorObsInfo):
        RoarPyActor.__init__(self, start_info.name if start_info.name is not None else "RoarPyRemoteClientActor", start_info.control_timestep)
        RoarPyObjectWithRemoteMessage.__init__(self)
        self.new_request_info = RoarPyRemoteActorObsInfoRequest(
            close=False,
            sensors_request={},
            action=None
        )
        self._closed = False
        self._next_action : typing.Optional[typing.Any] = None
        self._internal_sensors_map : typing.Dict[int, RoarPyRemoteClientSensor] = {}
        self._internal_action_spec : typing.Optional[gym.Space] = None
        self._depack_info(start_info)

    def _depack_info(self, data: RoarPyRemoteActorObsInfo) -> bool:
        self._control_timestep = data.control_timestep
        self._closed = data.is_closed
        for id, sensor_obs in data.sensors_map.items(): # Update sensors
            if id not in self._internal_sensors_map:
                self._internal_sensors_map[id] = RoarPyRemoteClientSensor(sensor_obs)
            else:
                self._internal_sensors_map[id]._depack_info(sensor_obs)
        for id, sensor in self._internal_sensors_map.items(): # Close sensors that are not in the new list
            if id not in data.sensors_map:
                sensor._closed = True
        
        self._refresh_sensor_list()
        new_action_spec = data.get_action_spec()
        if new_action_spec is not None:
            self._internal_action_spec = new_action_spec

    def _pack_info(self) -> RoarPyRemoteActorObsInfoRequest:
        self.new_request_info.sensors_request = {}
        for id, sensor in self._internal_sensors_map.items():
            self.new_request_info.sensors_request[id] = sensor._pack_info()
        if self._next_action is not None:
            next_action_flattened = gym.spaces.flatten(self._internal_action_spec, self._next_action)
            self.new_request_info.action = next_action_flattened
            self._next_action = None
        else:
            self.new_request_info.action = None
        return self.new_request_info
    
    def _refresh_sensor_list(self):
        to_del = []
        for id, sensor in self._internal_sensors_map.items():
            if sensor.is_closed():
                to_del.append(id)
        for id in to_del:
            del self._internal_sensors_map[id]

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        self._refresh_sensor_list()
        return self._internal_sensors_map.values()

    def get_action_spec(self) -> gym.Space:
        return self._internal_action_spec
    
    async def _apply_action(self, action: typing.Any) -> bool:
        self._next_action = action
        return True

    def close(self):
        self.new_request_info.close = True

    def is_closed(self) -> bool:
        return self._closed
    
@register_object_with_remote_message(RoarPyRemoteActorObsInfoRequest, RoarPyRemoteActorObsInfo)
class RoarPyRemoteServerActorPacker(RoarPyObjectWithRemoteMessage[RoarPyRemoteActorObsInfoRequest, RoarPyRemoteActorObsInfo]):
    def __init__(self, actor : RoarPyActor) -> None:
        super().__init__()

        self.actor = actor
        self.sensors_map : typing.Dict[int, RoarPyRemoteServerSensorWrapper] = {}
        self._last_sensor_id = 0
        self._refresh_sensor_list()
    
    def _refresh_sensor_list(self) -> None:
        new_map = {}
        actor_sensors = self.actor.get_sensors()
        for sensor in actor_sensors:
            found_in_map = False
            for oid, osensor in self.sensors_map:
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
            action_spec = self.actor.get_action_spec()
            try:
                real_action = gym.spaces.unflatten(action_spec, data.action)
            except:
                return False
            
            if action_spec.contains(real_action):
                self.actor.apply_action(real_action)
            else:
                return False
        
        if data.close:
            self.actor.close()
        
        self._refresh_sensor_list()
        for idx, sensor_req in data.sensors_request.items():
            if idx in self.sensors_map:
                self.sensors_map[idx]._depack_info(sensor_req)
        
        return True

    def _pack_info(self) -> RoarPyRemoteActorObsInfo:
        self._refresh_sensor_list()
        return RoarPyRemoteActorObsInfo(
            name=self.actor.name,
            control_timestep=self.actor.control_timestep,
            sensors_map={idx: sensor._pack_info() for idx, sensor in self.sensors_map.items()},
            is_closed=self.actor.is_closed(),
            action_spec=zlib.compress(pickle.dumps(self.actor.get_action_spec()))
        )