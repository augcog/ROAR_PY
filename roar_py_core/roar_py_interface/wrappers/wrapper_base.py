from typing import Any, TypeVar, Generic, Union, Callable, Iterable, Optional, Type, Dict, List
import threading

from roar_py_interface.worlds.waypoint import RoarPyWaypoint
from ..actors import RoarPyActor
from ..base.sensor import RoarPySensor
from ..worlds import RoarPyWorld, RoarPyWaypoint
import gymnasium as gym

_Wrapped = TypeVar("_Wrapped")
class RoarPyWrapper(Generic[_Wrapped]):
    def __init__(self, wrapped_object: Union[_Wrapped,"RoarPyWrapper[_Wrapped]"], wrapper_name: str):
        self._wrapped_object = wrapped_object
        self._wrapper_name = wrapper_name
    
    def __getattr__(self, __name: str) -> Any:
        if __name.startswith("_"):
            raise AttributeError("Cannot access private attribute")
        return getattr(self._wrapped_object, __name)
    
    def __str__(self) -> str:
        return self._wrapper_name + "(" + self._wrapped_object.__str__() + ")"

    @property
    def unwrapped(self) -> _Wrapped:
        if isinstance(self._wrapped_object, RoarPyWrapper):
            return self._wrapped_object.unwrapped
        else:
            return self._wrapped_object

class RoarPyActorWrapper(RoarPyWrapper[RoarPyActor], RoarPyActor):
    def __init__(self, wrapped_object: Union[RoarPyActor,RoarPyWrapper[RoarPyActor]], wrapper_name: str = "RoarPyActorWrapper"):
        RoarPyWrapper.__init__(self, wrapped_object, wrapper_name)
        RoarPyActor.__init__(self, wrapped_object.name, wrapped_object.control_timestep, wrapped_object.force_real_control_timestep)

    @property
    def control_timestep(self) -> float:
        return self._wrapped_object.control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value: float):
        self._wrapped_object.control_timestep = value
    
    @property
    def force_real_control_timestep(self) -> bool:
        return self._wrapped_object.force_real_control_timestep
    
    @force_real_control_timestep.setter
    def force_real_control_timestep(self, value: bool):
        self._wrapped_object.force_real_control_timestep = value

    @property
    def name(self) -> str:
        return self._wrapped_object.name
    
    @name.setter
    def name(self, value: str):
        self._wrapped_object.name = value

    def get_sensors(self) -> Iterable[RoarPySensor]:
        return self._wrapped_object.get_sensors()

    def get_action_spec(self) -> gym.Space:
        return self._wrapped_object.get_action_spec()
    
    async def _apply_action(self, action: Any) -> bool:
        return await self._wrapped_object._apply_action(action)

    def close(self):
        self._wrapped_object.close()

    def is_closed(self) -> bool:
        return self._wrapped_object.is_closed()

    async def apply_action(self, action: Any) -> bool:
        return await self._wrapped_object.apply_action(action)

_SensorWrapperObsT = TypeVar("_SensorWrapperObsT")
class RoarPySensorWrapper(Generic[_SensorWrapperObsT], RoarPyWrapper[RoarPySensor[_SensorWrapperObsT]], RoarPySensor[_SensorWrapperObsT]):
    def __init__(self, wrapped_object: Union[RoarPySensor[_SensorWrapperObsT], RoarPyWrapper[RoarPySensor[_SensorWrapperObsT]]], wrapper_name: str = "RoarPySensorWrapper"):
        RoarPyWrapper.__init__(self, wrapped_object, wrapper_name)
        RoarPySensor.__init__(self, wrapped_object.name, wrapped_object.control_timestep)
    
    @property
    def sensordata_type(self) -> Type:
        return self._wrapped_object.sensordata_type

    @property
    def control_timestep(self) -> float:
        return self._wrapped_object.control_timestep
    
    @control_timestep.setter
    def control_timestep(self, value: float):
        self._wrapped_object.control_timestep = value

    @property
    def name(self) -> str:
        return self._wrapped_object.name
    
    @name.setter
    def name(self, value: str):
        self._wrapped_object.name = value

    def get_gym_observation_spec(self) -> gym.Space:
        return self._wrapped_object.get_gym_observation_spec()
    
    async def receive_observation(self) -> _SensorWrapperObsT:
        return await self._wrapped_object.receive_observation()
    
    def get_last_observation(self) -> Optional[_SensorWrapperObsT]:
        return self._wrapped_object.get_last_observation()

    def convert_obs_to_gym_obs(self, obs: _SensorWrapperObsT):
        return self._wrapped_object.convert_obs_to_gym_obs(obs)
    
    def close(self):
        self._wrapped_object.close()
    
    def is_closed(self) -> bool:
        return self._wrapped_object.is_closed()

    def get_last_gym_observation(self) -> Optional[Any]:
        return self._wrapped_object.get_last_gym_observation()
    
class RoarPyWorldWrapper(RoarPyWrapper[RoarPyWorld], RoarPyWorld):
    def __init__(self, wrapped_object: Union[RoarPyWorld, RoarPyWrapper[RoarPyWorld]], wrapper_name: str = "RoarPyWorldWrapper"):
        RoarPyWrapper.__init__(self, wrapped_object, wrapper_name)
        RoarPyWorld.__init__(self)

    @property
    def is_asynchronous(self):
        return self._wrapped_object.is_asynchronous
    
    @is_asynchronous.setter
    def is_asynchronous(self, value: bool):
        self._wrapped_object.is_asynchronous = value
    
    def get_actors(self) -> Iterable[RoarPyActor]:
        return self._wrapped_object.get_actors()

    def get_sensors(self) -> Iterable[RoarPySensor]:
        return self._wrapped_object.get_sensors()
    
    @property
    def maneuverable_waypoints(self) -> Optional[Iterable[RoarPyWaypoint]]:
        return self._wrapped_object.maneuverable_waypoints

    @property
    def comprehensive_waypoints(self) -> Optional[Dict[Any, List[RoarPyWaypoint]]]:
        return self._wrapped_object.comprehensive_waypoints

    async def step(self) -> float:
        return await self._wrapped_object.step()
    
    @property
    def last_tick_elapsed_seconds(self) -> float:
        return self._wrapped_object.last_tick_elapsed_seconds

_TSWrapped = TypeVar("_TSWrapped")
class RoarPyThreadSafeWrapper(RoarPyWrapper[_TSWrapped], Generic[_TSWrapped]):
    def __init__(self, wrapped_object: Union[_TSWrapped,RoarPyWrapper[_TSWrapped]], tslock : threading.RLock):
        super().__init__(wrapped_object, wrapper_name="ThreadSafeWrapper")
        if not hasattr(self.unwrapped, "_tslock"):
            setattr(self.unwrapped, "_tslock", tslock)
        else:
            raise RuntimeError("Cannot wrap object twice")

def roar_py_thread_sync(func):
    def func_wrapper(self, *args, **kwargs):
        if hasattr(self, "_tslock"):
            with self._tslock:
                func_ret = func(self, *args, **kwargs)
                if isinstance(func_ret, RoarPyActor) or isinstance(func_ret, RoarPySensor) or isinstance(func_ret, RoarPyWorld):
                    func_ret = RoarPyThreadSafeWrapper(func_ret, self._tslock)
        else:
            func_ret = func(self, *args, **kwargs)
        return func_ret
    func_wrapper.is_thread_sync = True
    return func_wrapper

_ItemWrapped = TypeVar("_ItemWrapped")
class RoarPyAddItemWrapper(RoarPyWrapper[_ItemWrapped], Generic[_ItemWrapped]):
    def __init__(self, wrapped_object: Union[_ItemWrapped, RoarPyWrapper[_ItemWrapped]], add_callback : Callable[[Any], None] = None, remove_callback : Callable[[Any], None] = None):
        super().__init__(wrapped_object, wrapper_name="AddItemWrapper")
        if not hasattr(self.unwrapped, "_append_item_cb"):
            if add_callback is not None:
                setattr(self.unwrapped, "_append_item_cb", add_callback)
        else:
            raise RuntimeError("Cannot wrap object twice")
        
        if not hasattr(self.unwrapped, "_remove_item_cb"):
            if remove_callback is not None:
                setattr(self.unwrapped, "_remove_item_cb", remove_callback)
        else:
            raise RuntimeError("Cannot wrap object twice")

def roar_py_append_item(func):
    def func_wrapper(self, *args, **kwargs):
        func_ret = func(self, *args, **kwargs)
        if hasattr(self, "_append_item_cb"):
            self._append_item_cb(func_ret)
        return func_ret
    func_wrapper.is_append_item = True
    return func_wrapper

def roar_py_remove_item(func):
    def func_wrapper(self, arg1, *args, **kwargs):
        func_ret = func(self, arg1, *args, **kwargs)
        if hasattr(self, "_remove_item_cb"):
            self._remove_item_cb(arg1)
        return func_ret
    func_wrapper.is_remove_item = True
    return func_wrapper