from roar_py_interface import RoarPyWorld, RoarPySensor, RoarPyActor, RoarPyThreadSafeWrapper, RoarPyAddItemWrapper
import typing
import threading
import asyncio
import time
import rpyc

class RoarPyRemoteMaskedWorld(RoarPyWorld):
    def __init__(self, server_world : "RoarPyRemoteServer", shared_lock : threading.RLock):
        super().__init__()
        self.__shared_lock = shared_lock
        self.__server_world = server_world
        self._actors : typing.List[RoarPyActor] = []
        self._sensors : typing.List[RoarPySensor] = []
        self._ready_step = False
        self._last_step_dt = 0.0
    
    def __getattr__(self, __name: str):
        if __name.startswith("_"):
            raise AttributeError("Cannot access private attribute")
        attr_val = getattr(self.__server_world.__carla_world, __name)
        if callable(attr_val) and ((hasattr(attr_val, "is_append_item") and getattr(attr_val,"is_append_item")) or (hasattr(attr_val,"is_remove_item") and getattr(attr_val,"is_remove_item"))):
            def fn_wrapper(*args, **kwargs):
                with self.__shared_lock:
                    self.__server_world._last_subworld_modified = self
                return attr_val(*args, **kwargs)

            setattr(self, __name, fn_wrapper)
            return fn_wrapper
        else:
            return attr_val

    @property
    def is_asynchronous(self):
        return self.__server_world.is_asynchronous
    
    def get_actors(self) -> typing.Iterable[RoarPyActor]:
        return self._actors

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        return self._sensors

    async def step(self) -> float:
        with self.__shared_lock:
            self._ready_step = True
        while self._ready_step:
            await asyncio.sleep(0.001)
        ret = self._last_step_dt
        self._last_step_dt = 0.0
        return ret

    def close(self):
        with self.__shared_lock:
            for actor in self._actors:
                if not actor.is_closed():
                    actor.close()
            self._actors.clear()
            for sensor in self._sensors:
                if not sensor.is_closed():
                    sensor.close()
            self._sensors.clear()
            self.__server_world._masked_worlds.remove(self)

@rpyc.Service
class RoarPyRemoteServer:
    def __init__(
        self, 
        carla_world : RoarPyWorld, 
        is_asynchronous : bool,
        sync_wait_time_max : float = 5.0,
        constructor_to_subworld : typing.Callable[["RoarPyRemoteServer", threading.RLock], RoarPyRemoteMaskedWorld] = RoarPyRemoteMaskedWorld
    ):
        super().__init__()
        assert carla_world is not None and carla_world.is_asynchronous == False
        self.__shared_lock = threading.RLock()
        self.__carla_world = RoarPyAddItemWrapper(RoarPyThreadSafeWrapper(carla_world, self.__shared_lock), self.__add_item_callback, self.__remove_item_callback)
        self.__is_asynchronous = is_asynchronous
        self._masked_worlds : typing.List[RoarPyRemoteMaskedWorld] = []
        self._sync_wait_time_max = sync_wait_time_max
        self._constructor_to_subworld = constructor_to_subworld
        self._last_subworld_modified : RoarPyRemoteMaskedWorld = None

    @property
    def is_asynchronous(self):
        return self.__is_asynchronous

    def get_world(self) -> RoarPyRemoteMaskedWorld:
        with self.__shared_lock:
            new_masked_world = self._constructor_to_subworld(self, self.__shared_lock)
            self._masked_worlds.append(new_masked_world)
            return new_masked_world

    def __add_item_callback(self, item):
        if self._last_subworld_modified is None:
            return
        if isinstance(item, RoarPyActor):
            self._last_subworld_modified._actors.append(item)
        elif isinstance(item, RoarPySensor):
            self._last_subworld_modified._sensors.append(item)

    def __remove_item_callback(self, item):
        if self._last_subworld_modified is None:
            return
        if isinstance(item, RoarPyActor):
            self._last_subworld_modified._actors.remove(item)
        elif isinstance(item, RoarPySensor):
            self._last_subworld_modified._sensors.remove(item)

    def _step(self) -> float:
        if not self.is_asynchronous:
            not_ready_subworlds = self._masked_worlds.copy()
            start_time = time.time()
            while time.time() - start_time > self._sync_wait_time_max and len(not_ready_subworlds) > 0:
                not_ready_subworlds = filter(lambda x: x._ready_step == False, not_ready_subworlds)
            
            # if len(not_ready_subworlds) > 0:   
            #     print("Worlds not ready: ", not_ready_subworlds)
            #     print("Kicking them out of the list")
            #     for subworld in not_ready_subworlds:
            #         subworld.close()
            #         self._masked_worlds.remove(subworld)
        
        # Step the world
        with self.__shared_lock:
            stepped_dt = self.__carla_world.step()
            for masked_world in self._masked_worlds:
                masked_world._last_step_dt += stepped_dt
                masked_world._ready_step = False
            return stepped_dt