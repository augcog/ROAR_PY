from roar_py_interface.worlds import RoarPyWorld
from roar_py_interface.worlds.waypoint import RoarPyWaypoint
from roar_py_interface.wrappers.wrapper_base import RoarPyWrapper
from ..base import RoarPyObjectWithRemoteMessage, register_object_with_remote_message
from roar_py_interface import RoarPyWaypoint, RoarPyWorldWrapper, RoarPyWorld
from serde import serde
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Iterable, Any
from ..actors import RoarPyRemoteActorObsInfo, RoarPyRemoteActorObsInfoRequest, RoarPyRemoteServerActorWrapper, RoarPyRemoteClientActor
from ..sensors import RoarPyRemoteServerSensorWrapper, RoarPyRemoteClientSensor, RoarPyRemoteSensorObsInfo, RoarPyRemoteSensorObsInfoRequest
import asyncio
import copy

@serde
@dataclass
class RoarPyRemoteWorldInitInfo:
    maneuverable_waypoints: Optional[List[RoarPyWaypoint]]
    comprehensive_waypoints: Optional[Dict[Union[int, str, Any], List[RoarPyWaypoint]]]
    is_asynchronous: bool

@serde
@dataclass
class RoarPyRemoteWorldObsInfo:
    init_info: Optional[RoarPyRemoteWorldInitInfo]
    stepped: bool
    stepped_dt : float
    actor_info_map : Dict[int, RoarPyRemoteActorObsInfo]
    sensor_info_map : Dict[int, RoarPyRemoteSensorObsInfo]
    last_step_t : float

@serde
@dataclass
class RoarPyRemoteWorldObsInfoRequest:
    need_init_info: bool
    step: bool
    actor_request_map : Dict[int, RoarPyRemoteActorObsInfoRequest]
    sensor_request_map : Dict[int, RoarPyRemoteSensorObsInfoRequest]

@register_object_with_remote_message(RoarPyRemoteWorldObsInfoRequest, RoarPyRemoteWorldObsInfo)
class RoarPyRemoteServerWorldWrapper(RoarPyObjectWithRemoteMessage[RoarPyRemoteWorldObsInfoRequest, RoarPyRemoteWorldObsInfo], RoarPyWorldWrapper):
    def __init__(self, wrapped_object: Union[RoarPyWorld, RoarPyWorld]):
        RoarPyWorldWrapper.__init__(self, wrapped_object, "RoarPyRemoteWorldServerWrapper")
        RoarPyObjectWithRemoteMessage.__init__(self)
        self.actor_map : Dict[int, RoarPyRemoteServerActorWrapper] = {}
        self.sensor_map : Dict[int, RoarPyRemoteServerSensorWrapper] = {}
        self._last_actor_id = 0
        self._last_sensor_id = 0
        self._stepped = False
        self._stepped_dt = 0.0

        self._req_next_tick = False
        self._req_need_init_info = False
    
    async def step(self) -> float:
        dt = await RoarPyWorldWrapper.step(self)
        self._stepped = True
        self._stepped_dt = dt
    
    async def _tick_remote(self) -> None:
        if self._req_next_tick:
            self._req_next_tick = False
            await self.step()
        
        self._refresh_actor_list()
        self._refresh_sensor_list()

        all_sensor_and_actor_awaitables = []
        for actor in self.actor_map.values():
            all_sensor_and_actor_awaitables.append(actor._tick_remote())
        for sensor in self.sensor_map.values():
            all_sensor_and_actor_awaitables.append(sensor._tick_remote())
        await asyncio.gather(*all_sensor_and_actor_awaitables)

    def _refresh_actor_list(self) -> None:
        new_map = {}
        actor_list = self._wrapped_object.get_actors()
        for actor in actor_list:
            found_in_map = False
            for oid, oactor in self.actor_map.items():
                if actor is oactor._wrapped_object:
                    new_map[oid] = oactor
                    found_in_map = True
                    break
            if not found_in_map:
                new_map[self._last_actor_id] = RoarPyRemoteServerActorWrapper(actor)
                self._last_actor_id += 1
        self.actor_map = new_map
    
    def _refresh_sensor_list(self) -> None:
        new_map = {}
        sensor_list = self._wrapped_object.get_sensors()
        for sensor in sensor_list:
            found_in_map = False
            for oid, osensor in self.sensor_map.items():
                if sensor is osensor._wrapped_object:
                    new_map[oid] = osensor
                    found_in_map = True
                    break
            if not found_in_map:
                new_map[self._last_sensor_id] = RoarPyRemoteServerSensorWrapper(sensor)
                self._last_sensor_id += 1
        self.sensor_map = new_map
    
    def _depack_info(self, data: RoarPyRemoteWorldObsInfoRequest) -> bool:
        self._stepped = False
        self._stepped_dt = 0.0
        self._req_need_init_info = data.need_init_info
        self._req_next_tick = data.step
        self._refresh_actor_list()
        for oid, actor_request in data.actor_request_map.items():
            if oid in self.actor_map:
                self.actor_map[oid]._depack_info(actor_request)
        self._refresh_sensor_list()
        for oid, sensor_request in data.sensor_request_map.items():
            if oid in self.sensor_map:
                self.sensor_map[oid]._depack_info(sensor_request)
        return True
    
    def _pack_info(self) -> RoarPyRemoteWorldObsInfo:
        self._refresh_actor_list()
        self._refresh_sensor_list()
        actor_info_map = {}
        for oid, actor in self.actor_map.items():
            actor_info_map[oid] = actor._pack_info()
        sensor_info_map = {}
        for oid, sensor in self.sensor_map.items():
            sensor_info_map[oid] = sensor._pack_info()
        ret = RoarPyRemoteWorldObsInfo(
            init_info=None if not self._req_need_init_info else RoarPyRemoteWorldInitInfo(
                maneuverable_waypoints=list(self.maneuverable_waypoints),
                comprehensive_waypoints=self.comprehensive_waypoints,
                is_asynchronous=self.is_asynchronous
            ),
            stepped=self._stepped,
            stepped_dt=self._stepped_dt,
            actor_info_map=actor_info_map,
            sensor_info_map=sensor_info_map,
            last_step_t=self.last_tick_elapsed_seconds
        )
        # self._stepped = False
        return ret

@register_object_with_remote_message(RoarPyRemoteWorldObsInfo, RoarPyRemoteWorldObsInfoRequest)
class RoarPyRemoteClientWorld(RoarPyWorld, RoarPyObjectWithRemoteMessage[RoarPyRemoteWorldObsInfo, RoarPyRemoteWorldObsInfoRequest]):
    def __init__(self, start_info : RoarPyRemoteActorObsInfo) -> None:
        RoarPyWorld.__init__(self)
        RoarPyObjectWithRemoteMessage.__init__(self)
        self._actor_map : Dict[int, RoarPyRemoteClientActor] = {}
        self._sensor_map : Dict[int, RoarPyRemoteClientSensor] = {}
        self._is_asynchronous = False
        self._last_step_t = 0.0
        self._maneuverable_waypoints = None
        self._comprehensive_waypoints = None

        self._new_info : Optional[RoarPyRemoteWorldObsInfo] = None

        self._req_need_init_info = True

        self._depack_info(start_info)
    
    @property
    def is_asynchronous(self):
        return self._is_asynchronous
    
    def get_actors(self) -> Iterable[RoarPyRemoteClientActor]:
        return list(self._actor_map.values())

    def get_sensors(self) -> Iterable[RoarPyRemoteClientSensor]:
        return list(self._sensor_map.values())
    
    @property
    def maneuverable_waypoints(self) -> Optional[Iterable[RoarPyWaypoint]]:
        return self._maneuverable_waypoints

    @property
    def comprehensive_waypoints(self) -> Optional[Dict[Any, List[RoarPyWaypoint]]]:
        return self._comprehensive_waypoints

    def _update_actor_map(self, actor_info_map : Dict[int, RoarPyRemoteActorObsInfo]) -> None:
        for oid, actor_info in actor_info_map.items():
            if oid not in self._actor_map:
                self._actor_map[oid] = RoarPyRemoteClientActor(actor_info)
        for oid in self._actor_map.keys():
            if oid not in actor_info_map:
                del self._actor_map[oid]
        
    def _update_sensor_map(self, sensor_info_map : Dict[int, RoarPyRemoteSensorObsInfo]) -> None:
        for oid, sensor_info in sensor_info_map.items():
            if oid not in self._sensor_map:
                tmp_sensor_info = copy.copy(sensor_info)
                tmp_sensor_info.last_data = None
                self._sensor_map[oid] = RoarPyRemoteClientSensor(tmp_sensor_info)
        for oid in self._sensor_map.keys():
            if oid not in sensor_info_map:
                del self._sensor_map[oid]

    def _depack_actor_infos(self, actor_info_map : Dict[int, RoarPyRemoteActorObsInfo]) -> None:
        for oid, actor_info in actor_info_map.items():
            if oid in self._actor_map:
                self._actor_map[oid]._depack_info(actor_info)
            else:
                self._actor_map[oid] = RoarPyRemoteClientActor(actor_info)
    
    def _depack_sensor_infos(self, sensor_info_map : Dict[int, RoarPyRemoteSensorObsInfo]) -> None:
        for oid, sensor_info in sensor_info_map.items():
            if oid in self._sensor_map:
                self._sensor_map[oid]._depack_info(sensor_info)
            else:
                self._sensor_map[oid] = RoarPyRemoteClientSensor(sensor_info)

    async def step(self) -> float:
        self._recv_stepped = False
        self._recv_last_stepped_dt = 0.0

        if self._new_info is None or not self._new_info.stepped:
            return 0.0 # Potentially we can also raise error

        self._depack_actor_infos(self._new_info.actor_info_map)
        self._depack_sensor_infos(self._new_info.sensor_info_map)
        self._last_step_t = self._new_info.last_step_t

        dt = self._new_info.stepped_dt
        self._new_info = None # Clear the new info after we have processed it, so that we can receive new info by stepping on the server side
        return dt
    
    @property
    def last_tick_elapsed_seconds(self) -> float:
        return self._last_step_t

    def _depack_info(self, data: RoarPyRemoteWorldObsInfo) -> bool:
        if data.init_info is not None:
            self._maneuverable_waypoints = data.init_info.maneuverable_waypoints
            self._comprehensive_waypoints = data.init_info.comprehensive_waypoints
            self._is_asynchronous = data.init_info.is_asynchronous
            self._req_need_init_info = False
        
        self._update_actor_map(data.actor_info_map)
        self._update_sensor_map(data.sensor_info_map)

        if self._new_info is None:
            self._new_info = data
            self._new_info.init_info = None
            if not self._new_info.stepped:
                self._new_info.stepped_dt = 0.0
        else:
            if data.stepped:
                self._new_info.stepped = True
                self._new_info.stepped_dt += max(data.stepped_dt, 0.0)
            # Overwrite the actor and sensor info map with the new one
            self._new_info.actor_info_map = data.actor_info_map
            self._new_info.sensor_info_map = data.sensor_info_map
    
    def _pack_info(self) -> RoarPyRemoteWorldObsInfoRequest:
        return RoarPyRemoteWorldObsInfoRequest(
            need_init_info=self._req_need_init_info,
            step=True, # Always step on the server side
            actor_request_map={oid: actor._pack_info() for oid, actor in self._actor_map.items()},
            sensor_request_map={oid: sensor._pack_info() for oid, sensor in self._sensor_map.items()},
        )