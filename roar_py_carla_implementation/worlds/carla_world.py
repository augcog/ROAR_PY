from roar_py_interface import RoarPyWorld
import carla
import typing
import asyncio
import numpy as np

from roar_py_interface import RoarPyActor, RoarPySensor
from ..actors import RoarPyCarlaVehicle

class RoarPyCarlaWorld(RoarPyWorld):
    ASYNC_SLEEP_TIME = 0.005

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
    def is_asynchronous(self):
        native_settings = self.get_native_settings()
        return not native_settings.synchronous_mode
    
    def set_asynchronous(self, asynchronous : bool):
        self.setup_mode(asynchronous)
        native_settings = self.get_native_settings()
        native_settings.synchronous_mode = not asynchronous
        self.carla_world.apply_settings(native_settings)
    
    def setup_mode(self, target_asynchronous : bool):
        if target_asynchronous:
            if self.tick_callback_id is not None:
                return
            self.tick_callback_id = self.carla_world.on_tick(self.on_tick_recv)
        else:
            if self.tick_callback_id is not None:
                self.carla_world.remove_on_tick(self.tick_callback_id)
            self.tick_callback_id = None
    
    # Time between two server ticks in seconds, 0.0 means variable timestep
    @property
    def control_timestep(self) -> float:
        return self.carla_world.get_settings().fixed_delta_seconds
    
    @control_timestep.setter
    def control_timestep(self, control_timestep : float):
        native_settings = self.carla_world.get_settings()
        native_settings.fixed_delta_seconds = control_timestep
        native_settings.max_substep_delta_time = control_timestep / native_settings.max_substeps
        self.carla_world.apply_settings(native_settings)
    
    @property
    def contorl_subtimestep(self) -> float:
        return self.carla_world.get_settings().max_substep_delta_time

    @contorl_subtimestep.setter
    def contorl_subtimestep(self, contorl_subtimestep : float):
        native_settings = self.carla_world.get_settings()
        assert contorl_subtimestep < native_settings.fixed_delta_seconds
        assert native_settings.fixed_delta_seconds % contorl_subtimestep == 0
        native_settings.max_substeps = int(native_settings.fixed_delta_seconds / contorl_subtimestep)
        native_settings.max_substep_delta_time = contorl_subtimestep
        self.carla_world.apply_settings(native_settings)
    
    def on_tick_recv(self, world_snapshot : carla.WorldSnapshot):
        self.last_tick_time = world_snapshot.timestamp.elapsed_seconds
    
    async def step(self) -> float:
        if self.is_asynchronous():
            start_time = self.last_tick_time
            while self.last_tick_time == start_time:
                await asyncio.sleep(__class__.ASYNC_SLEEP_TIME)
                # Instead of waitForTick, we use sleep here to avoid blocking the event loop
            
            if start_time is None:
                return 0.0
            else:
                return self.last_tick_time - start_time
        else:
            self.carla_world.tick(seconds=60.0) # server waits 60s for client to finish the tick
            return self.control_timestep
    
    def get_weather(self) -> carla.WeatherParameters:
        return self.carla_world.get_weather()
    
    def set_weather(self, weather : carla.WeatherParameters):
        self.carla_world.set_weather(weather)

    def attach_native_carla_actor(
        self,
        blueprint_id : str, 
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray
    ) -> carla.Actor:
        assert location.shape == (3,) and roll_pitch_yaw.shape == (3,)
        blueprint = self.carla_world.get_blueprint_library().find(blueprint_id)
        transform = carla.Transform(carla.Location(*location), carla.Rotation(roll=roll_pitch_yaw[0], pitch=roll_pitch_yaw[1], yaw=roll_pitch_yaw[2]))
        new_actor = self.carla_world.spawn_actor(blueprint, transform, attach_to=None, attachment=carla.AttachmentType.Rigid)
        return new_actor

    """
    Spawn a vehicle in the world
    See https://carla.readthedocs.io/en/latest/bp_library/#vehicle for blueprint_ids
    """
    def spawn_vehicle(
        self,
        blueprint_id : str,
        location : np.ndarray,
        roll_pitch_yaw : np.ndarray,
        auto_gear: bool = True,
        name: str = "carla_vehicle"
    ):
        new_actor = self.attach_native_carla_actor(blueprint_id, location, roll_pitch_yaw)
        new_vehicle = RoarPyCarlaVehicle(self.carla_instance, new_actor, auto_gear, name=name)
        self._actors.append(new_vehicle)
        return new_vehicle