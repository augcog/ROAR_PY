from roar_py_interface import RoarPyWorld, RoarPySensor, RoarPyActor, RoarPyThreadSafeWrapper, RoarPyAddItemWrapper, RoarPyWaypoint, RoarPyWrapper
import typing
import threading
import asyncio
import time
import weakref

class RoarPyRemoteMaskedWorld(RoarPyWorld):
    def __init__(self, server_world : "RoarPyRemoteServerWorldManager", underlying_world : RoarPyWorld, shared_lock : threading.RLock):
        super().__init__()
        self.__shared_lock = shared_lock
        self.__server_world = server_world
        self.__underlying_world = underlying_world
        self._actors : typing.List[RoarPyActor] = []
        self._sensors : typing.List[RoarPySensor] = []
        self._ready_step = False
        self._next_step_dt = 0.0
        self._last_step_timestamp = underlying_world.last_tick_elapsed_seconds
        self._next_step_timestamp = self._last_step_timestamp
    
    def __getattr__(self, __name: str):
        if __name.startswith("_"):
            raise AttributeError("Cannot access private attribute")
        attr_val = getattr(self.__underlying_world, __name)
        if callable(attr_val) and ((hasattr(attr_val, "is_append_item") and getattr(attr_val,"is_append_item")) or (hasattr(attr_val,"is_remove_item") and getattr(attr_val,"is_remove_item"))):
            def fn_wrapper(*args, **kwargs):
                with self.__shared_lock:
                    self.__server_world._last_subworld_modified = weakref.proxy(self, self.__server_world._del_masked_world)
                return attr_val(*args, **kwargs)

            setattr(self, __name, fn_wrapper)
            return fn_wrapper
        else:
            return attr_val

    @property
    def is_asynchronous(self):
        return self.__server_world.is_asynchronous
    
    def _refresh_actor_list(self):
        new_actor_list = []
        for actor in self._actors:
            if not actor.is_closed():
                new_actor_list.append(actor)
        self._actors = new_actor_list
    
    def _refresh_sensor_list(self):
        new_sensor_list = []
        for sensor in self._sensors:
            if not sensor.is_closed():
                new_sensor_list.append(sensor)
        self._sensors = new_sensor_list

    def get_actors(self) -> typing.Iterable[RoarPyActor]:
        self._refresh_actor_list()
        return self._actors.copy()

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        self._refresh_sensor_list()
        return self._sensors.copy()

    @property
    def maneuverable_waypoints(self) -> typing.Optional[typing.Iterable[RoarPyWaypoint]]:
        return self.__underlying_world.maneuverable_waypoints

    @property
    def comprehensive_waypoints(self) -> typing.Optional[typing.Dict[typing.Any, typing.List[RoarPyWaypoint]]]:
        return self.__underlying_world.comprehensive_waypoints

    @property
    def last_tick_elapsed_seconds(self) -> float:
        return self._last_step_timestamp

    async def step(self) -> float:
        with self.__shared_lock:
            self._ready_step = True
        # while self._ready_step:
        #     await asyncio.sleep(0.1)
        ret = self._next_step_dt
        self._last_step_timestamp = self._next_step_timestamp
        self._next_step_dt = 0.0
        return ret

    def close(self):
        try:
            with self.__shared_lock:
                for actor in self._actors:
                    if not actor.is_closed():
                        actor.close()
                self._actors.clear()
                for sensor in self._sensors:
                    if not sensor.is_closed():
                        sensor.close()
                self._sensors.clear()
                if self in self.__server_world._masked_worlds:
                    self.__server_world._del_masked_world(self)
        except:
            pass
    
    def __del__(self):
        self.close()

class RoarPyRemoteServerWorldManager:
    def __init__(
        self, 
        world : RoarPyWorld, 
        is_asynchronous : bool,
        sync_wait_time_max : float = 5.0,
        constructor_to_subworld : typing.Callable[["RoarPyRemoteServerWorldManager", RoarPyWorld, threading.RLock], RoarPyRemoteMaskedWorld] = RoarPyRemoteMaskedWorld,
        thread_safe = True
    ):
        super().__init__()
        assert world is not None
        self.__shared_lock = threading.RLock()

        ts_world = RoarPyThreadSafeWrapper(world, self.__shared_lock) if thread_safe else world
        self.__underlying_world : RoarPyWorld = RoarPyAddItemWrapper(ts_world, self.__add_item_callback, self.__remove_item_callback)
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
            new_masked_world = self._constructor_to_subworld(self, self.__underlying_world, self.__shared_lock)
            self._masked_worlds.append(weakref.proxy(new_masked_world,self._del_masked_world))
            print("Creating new masked world => new masked world count {}".format(len(self._masked_worlds)))
            return new_masked_world
        
    def _del_masked_world(self,world : RoarPyRemoteMaskedWorld):
        with self.__shared_lock:
            try:
                if world in self._masked_worlds:
                    self._masked_worlds.remove(world)
                    print("Deleted masked world => new masked world count {}".format(len(self._masked_worlds)))
                if self._last_subworld_modified is world:
                    self._last_subworld_modified = None
            except:
                pass

    def __add_item_callback(self, item):
        if self._last_subworld_modified is None:
            return
        
        item_unwrapped = item if not isinstance(item, RoarPyWrapper) else item.unwrapped
        
        if isinstance(item_unwrapped, RoarPyActor):
            # actors_in_underlying_world = self.__underlying_world.get_actors()
            # unwrapped_actors_in_underlying_world = [actor if not isinstance(actor, RoarPyWrapper) else actor.unwrapped for actor in actors_in_underlying_world]
            # if item_unwrapped in unwrapped_actors_in_underlying_world:
            self._last_subworld_modified._actors.append(item)
        elif isinstance(item_unwrapped, RoarPySensor):
            sensors_in_underlying_world = self.__underlying_world.get_sensors()
            unwrapped_sensors_in_underlying_world = [sensor if not isinstance(sensor, RoarPyWrapper) else sensor.unwrapped for sensor in sensors_in_underlying_world]
            if item_unwrapped in unwrapped_sensors_in_underlying_world:
                self._last_subworld_modified._sensors.append(item)

    def __remove_item_callback(self, item):
        if self._last_subworld_modified is None:
            return
        if isinstance(item, RoarPyActor):
            if item in self._last_subworld_modified._actors:
                self._last_subworld_modified._actors.remove(item)
        elif isinstance(item, RoarPySensor):
            if item in self._last_subworld_modified._sensors:
                self._last_subworld_modified._sensors.remove(item)

    async def _step(self) -> float:
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
            stepped_dt = await self.__underlying_world.step()
            for masked_world in self._masked_worlds:
                masked_world._next_step_dt += stepped_dt
                masked_world._ready_step = False
                masked_world._next_step_timestamp = self.__underlying_world.last_tick_elapsed_seconds
            return stepped_dt