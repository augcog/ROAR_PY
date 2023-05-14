from roar_py_interface import RoarPyWorld
import carla
import typing
import asyncio
import numpy as np

from roar_py_interface import RoarPyActor, RoarPySensor, roar_py_thread_sync, roar_py_append_item, roar_py_remove_item
from ..actors import RoarPyCarlaVehicle
from functools import cached_property

class RoarPyCarlaWorld(RoarPyWorld):
    ASYNC_SLEEP_TIME = 0.005
    WAYPOINTS_DISTANCE = 1.0

    def __init__(
        self,
        carla_world : carla.World,
        carla_instance: "RoarPyCarlaInstance"
    ) -> None:
        super().__init__()
        self.carla_world = carla_world
        self.carla_instance = carla_instance
        self.tick_callback_id : typing.Optional[int] = None
        self.last_tick_time : typing.Optional[float] = None
        self.setup_mode(self.is_asynchronous)
        self._actors : typing.List[RoarPyActor] = []

    def get_actors(self) -> typing.Iterable[RoarPyActor]:
        return self._actors.copy()

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        return []
        
    def get_all_sensors(self) -> typing.Iterable[RoarPySensor]:
        for carlabase in self.carla_instance.actor_to_instance_map.values():
            if isinstance(carlabase, RoarPySensor):
                yield carlabase

    @property
    @roar_py_thread_sync
    def is_asynchronous(self):
        native_settings = self.carla_world.get_settings()
        return not native_settings.synchronous_mode
    
    @roar_py_thread_sync
    def set_asynchronous(self, asynchronous : bool):
        self.setup_mode(asynchronous)
        native_settings = self.carla_world.get_settings()
        native_settings.synchronous_mode = not asynchronous
        self.carla_world.apply_settings(native_settings)
    
    @roar_py_thread_sync
    def setup_mode(self, target_asynchronous : bool):
        if target_asynchronous:
            if self.tick_callback_id is not None:
                return
            self.tick_callback_id = self.carla_world.on_tick(self.__on_tick_recv)
        else:
            if self.tick_callback_id is not None:
                self.carla_world.remove_on_tick(self.tick_callback_id)
            self.tick_callback_id = None
    
    # Time between two server ticks in seconds, 0.0 means variable timestep
    @property
    def control_timestep(self) -> float:
        return self.carla_world.get_settings().fixed_delta_seconds
    
    @control_timestep.setter
    @roar_py_thread_sync
    def control_timestep(self, control_timestep : float):
        native_settings = self.carla_world.get_settings()
        native_settings.fixed_delta_seconds = control_timestep
        native_settings.max_substep_delta_time = control_timestep / native_settings.max_substeps
        self.carla_world.apply_settings(native_settings)
    
    @property
    def contorl_subtimestep(self) -> float:
        return self.carla_world.get_settings().max_substep_delta_time

    @contorl_subtimestep.setter
    @roar_py_thread_sync
    def contorl_subtimestep(self, contorl_subtimestep : float):
        native_settings = self.carla_world.get_settings()
        assert contorl_subtimestep < native_settings.fixed_delta_seconds
        assert native_settings.fixed_delta_seconds % contorl_subtimestep == 0
        native_settings.max_substeps = int(native_settings.fixed_delta_seconds / contorl_subtimestep)
        native_settings.max_substep_delta_time = contorl_subtimestep
        self.carla_world.apply_settings(native_settings)
    
    @roar_py_thread_sync
    def set_control_steps(self, control_timestep : float, control_substimestep : float):
        assert control_substimestep <= control_timestep and control_timestep % control_substimestep < 1e-6
        native_settings = self.carla_world.get_settings()
        native_settings.fixed_delta_seconds = control_timestep
        native_settings.max_substeps = int(control_timestep / control_substimestep)
        native_settings.max_substep_delta_time = control_substimestep
        self.carla_world.apply_settings(native_settings)

    def __on_tick_recv(self, world_snapshot : carla.WorldSnapshot):
        self.last_tick_time = world_snapshot.timestamp.elapsed_seconds
    
    @roar_py_thread_sync
    async def step(self) -> float:
        if self.is_asynchronous:
            start_time = self.last_tick_time
            while self.last_tick_time == start_time:
                await asyncio.sleep(__class__.ASYNC_SLEEP_TIME)
                # Instead of waitForTick, we use sleep here to avoid blocking the event loop
            
            if start_time is None:
                dt = 0.0
            else:
                dt = self.last_tick_time - start_time
            return dt
        else:
            self.carla_world.tick(seconds=60.0) # server waits 60s for client to finish the tick
            return self.control_timestep
    
    @roar_py_thread_sync
    def _get_weather(self) -> carla.WeatherParameters:
        return self.carla_world.get_weather()
    
    @roar_py_thread_sync
    def _set_weather(self, weather : carla.WeatherParameters):
        self.carla_world.set_weather(weather)

    def find_blueprint(self, id: str) -> carla.ActorBlueprint:
        return self.carla_world.get_blueprint_library().find(id)
    
    """
    Get a list of all available spawn points
    Output: [(location, rotation), ...]
    rotation is [roll, pitch, yaw] in radians
    """
    @cached_property
    def spawn_points(self) -> typing.List[typing.Tuple[np.ndarray, np.ndarray]]:
        native_spawn_points = self._native_carla_map.get_spawn_points()
        ret = []
        for native_spawn_point in native_spawn_points:
            location = np.array([native_spawn_point.location.x, native_spawn_point.location.y, native_spawn_point.location.z])
            rotation = np.deg2rad(np.array([native_spawn_point.rotation.roll, native_spawn_point.rotation.pitch, native_spawn_point.rotation.yaw]))
            ret.append((location, rotation))
        return ret

    """
    Convert a location in world coordinate to geolocation
    Output: [latitude (deg), longitude (deg), altitude(m)]
    """
    def get_geolocation(self, location : np.ndarray) -> np.ndarray:
        assert location.shape == (3,)
        native_location = carla.Location(x=location[0], y=location[1], z=location[2])
        native_geolocation = self._native_carla_map.transform_to_geolocation(native_location)
        return np.array([native_geolocation.latitude, native_geolocation.longitude, native_geolocation.altitude])

    @cached_property
    @roar_py_thread_sync
    def _native_carla_waypoints(self) -> typing.List[carla.Waypoint]:
        return self._native_carla_map.generate_waypoints(__class__.WAYPOINTS_DISTANCE)

    @cached_property
    @roar_py_thread_sync
    def _native_carla_map(self) -> carla.Map:
        return self.carla_world.get_map()

    @roar_py_thread_sync
    def _attach_native_carla_actor(
        self,
        blueprint : carla.ActorBlueprint, 
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray
    ) -> carla.Actor:
        assert location.shape == (3,) and roll_pitch_yaw.shape == (3,)
        location = location.astype(float)
        roll_pitch_yaw = np.deg2rad(roll_pitch_yaw).astype(float)

        transform = carla.Transform(carla.Location(*location), carla.Rotation(roll=roll_pitch_yaw[0], pitch=roll_pitch_yaw[1], yaw=roll_pitch_yaw[2]))
        new_actor = self.carla_world.try_spawn_actor(blueprint, transform, None, carla.AttachmentType.Rigid)
        return new_actor

    """
    Spawn a vehicle in the world
    See https://carla.readthedocs.io/en/latest/bp_library/#vehicle for blueprint_ids
    """
    @roar_py_append_item
    @roar_py_thread_sync
    def spawn_vehicle(
        self,
        blueprint_id : str,
        location : np.ndarray,
        roll_pitch_yaw : np.ndarray,
        auto_gear: bool = True,
        name: str = "carla_vehicle",
        rgba: typing.Optional[np.ndarray] = None
    ) -> typing.Optional[RoarPyCarlaVehicle]:
        blueprint = self.find_blueprint(blueprint_id)
        if blueprint is None:
            return None
        if not blueprint.has_tag("vehicle"):
            return None
        
        if rgba is not None and rgba.shape == (4,):
            vehicle_color = carla.Color(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            blueprint.set_attribute("color", str(vehicle_color))

        new_actor = self._attach_native_carla_actor(blueprint, location, roll_pitch_yaw)
        if new_actor is None:
            return None
        
        new_vehicle = RoarPyCarlaVehicle(self.carla_instance, new_actor, auto_gear, name=name)
        self._actors.append(new_vehicle)
        return new_vehicle
    
    @roar_py_remove_item
    @roar_py_thread_sync
    def remove_actor(self, actor : RoarPyActor):
        self._actors.remove(actor)
        actor.close()