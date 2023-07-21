from roar_py_interface import RoarPyWorld
import carla
from ..carla_agents.navigation.global_route_planner import GlobalRoutePlanner as CarlaGlobalRoutePlanner
import typing
import asyncio
import numpy as np
import os.path

from roar_py_interface import RoarPyActor, RoarPySensor, roar_py_thread_sync, roar_py_append_item, roar_py_remove_item, RoarPyWaypoint
from roar_py_interface.sensors import *
from ..actors import RoarPyCarlaVehicle, RoarPyCarlaActor
from ..sensors import *
from functools import cached_property
import networkx as nx
from ..utils import *
import transforms3d as tr3d

class RoarPyCarlaWorld(RoarPyWorld):
    ASYNC_SLEEP_TIME = 0.005
    WAYPOINTS_DISTANCE = 1.0
    ASSET_DIR = os.path.dirname(os.path.dirname(__file__)) + "/assets"

    def __init__(
        self,
        carla_world : carla.World,
        carla_instance: "RoarPyCarlaInstance"
    ) -> None:
        super().__init__()
        self.carla_world = carla_world
        self.carla_instance = carla_instance
        self.tick_callback_id : typing.Optional[int] = None
        self._last_tick_time : float = 0.0
        self._actors : typing.List[RoarPyCarlaActor] = []
        self._sensors : typing.List[RoarPySensor] = []

        carla_settings = carla_world.get_settings()
        self._control_timestep = carla_settings.fixed_delta_seconds
        self._control_subtimestep = carla_settings.max_substep_delta_time
        self._is_asynchronous = not carla_settings.synchronous_mode
        self.setup_mode(self.is_asynchronous)

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
        
    def get_all_sensors(self) -> typing.Iterable[RoarPySensor]:
        for carlabase in self.carla_instance.actor_to_instance_map.values():
            if isinstance(carlabase, RoarPySensor):
                yield carlabase

    @property
    def comprehensive_waypoints(self) -> typing.Dict[typing.Any,typing.List[RoarPyWaypoint]]:
        ret = {}
        for waypoint in self._native_carla_waypoints:
            transform_w = transform_from_carla(waypoint.transform)
            new_waypoint = RoarPyWaypoint(
                transform_w[0],
                transform_w[1],
                waypoint.lane_width
            )
            if waypoint.road_id not in ret:
                ret[waypoint.road_id] = []
            ret[waypoint.road_id].append(new_waypoint)
        return ret

    @cached_property
    @roar_py_thread_sync
    def maneuverable_waypoints(self) -> typing.Optional[typing.List[RoarPyWaypoint]]:
        waypoint_asset_dir = __class__.ASSET_DIR + "/waypoints"
        waypoint_file = waypoint_asset_dir + "/" + self.map_name + ".npz"
        if os.path.exists(waypoint_file):
            way_points = np.load(waypoint_file)
            return RoarPyWaypoint.load_waypoint_list(way_points)
        
        spawn_points = self.spawn_points
        num_spawn_points = len(spawn_points)

        waypoints = None
        if num_spawn_points > 1:
            waypoints = []
            for i in range(num_spawn_points):
                curr_start = spawn_points[i][0]
                curr_end = spawn_points[(i+1)%num_spawn_points][0]
                # print("Tracing from {} to {}".format(curr_start, curr_end))
                current_traced = self._trace_waypoint(
                    curr_start,
                    curr_end
                )
                if current_traced is not None:
                    waypoints.extend(current_traced)
                else:
                    waypoints = None
                    break
                
        if num_spawn_points == 1 or (num_spawn_points > 1 and waypoints is None):
            init_pos = spawn_points[0][0]
            init_rot = spawn_points[0][1]

            pos_y = np.array([1.0,0.0,0.0]) # Go forward 1m and set as next waypoint

            second_pos = init_pos + tr3d.euler.euler2mat(*init_rot).dot(pos_y)
            first_to_second = self._trace_waypoint(init_pos, second_pos)
            second_to_first = self._trace_waypoint(second_pos, init_pos)
            if first_to_second is not None and second_to_first is not None:
                waypoints = list(first_to_second) + list(second_to_first)
            else:
                waypoints = None
            
        return waypoints

    @cached_property
    @roar_py_thread_sync
    def _native_route_tracer(self):
        return CarlaGlobalRoutePlanner(self._native_carla_map, self.WAYPOINTS_DISTANCE)

    def _trace_waypoint(self, from_location : np.ndarray, to_location : np.ndarray) -> typing.Optional[typing.Iterable[RoarPyWaypoint]]:
        location_carla = location_to_carla(from_location)
        destination_carla = location_to_carla(to_location)
        try:
            native_waypoints = self._native_route_tracer.trace_route(location_carla, destination_carla)
        except nx.NetworkXNoPath:
            return None
        for native_waypoint, native_waypoint_road_option in native_waypoints:
            transform_w = transform_from_carla(native_waypoint.transform)
            # Interestingly carla's waypoint also has x axis pointing to the "forward" of the road
            real_w = RoarPyWaypoint(
                transform_w[0],
                transform_w[1],
                native_waypoint.lane_width
            )
            yield real_w

    @cached_property
    def map_name(self):
        native_name = self._native_carla_map.name
        if "/" in native_name:
            return native_name.split("/")[-1]
        else:
            return native_name

    @property
    @roar_py_thread_sync
    def is_asynchronous(self):
        return self._is_asynchronous
    
    @roar_py_thread_sync
    def set_asynchronous(self, asynchronous : bool):
        self.setup_mode(asynchronous)
        native_settings = self.carla_world.get_settings()
        native_settings.synchronous_mode = not asynchronous
        self.carla_world.apply_settings(native_settings)
        self._is_asynchronous = asynchronous
    
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
        return self._control_timestep
    
    @control_timestep.setter
    @roar_py_thread_sync
    def control_timestep(self, control_timestep : float):
        native_settings = self.carla_world.get_settings()
        native_settings.fixed_delta_seconds = control_timestep
        native_settings.max_substep_delta_time = control_timestep / native_settings.max_substeps
        self.carla_world.apply_settings(native_settings)
        self._control_timestep = control_timestep
    
    @property
    def contorl_subtimestep(self) -> float:
        return self._control_subtimestep

    @contorl_subtimestep.setter
    @roar_py_thread_sync
    def contorl_subtimestep(self, contorl_subtimestep : float):
        native_settings = self.carla_world.get_settings()
        assert contorl_subtimestep < native_settings.fixed_delta_seconds
        assert native_settings.fixed_delta_seconds % contorl_subtimestep == 0
        native_settings.max_substeps = int(native_settings.fixed_delta_seconds / contorl_subtimestep)
        native_settings.max_substep_delta_time = contorl_subtimestep
        self.carla_world.apply_settings(native_settings)
        self._control_subtimestep = contorl_subtimestep
    
    @roar_py_thread_sync
    def set_control_steps(self, control_timestep : float, control_substimestep : float):
        assert (
            (control_substimestep <= control_timestep and control_timestep % control_substimestep < 1e-6) or 
            (control_timestep <= 0)
        )
        native_settings = self.carla_world.get_settings()
        native_settings.fixed_delta_seconds = control_timestep if control_timestep > 0 else None
        native_settings.max_substeps = int(control_timestep / control_substimestep) if control_timestep > 0 else 100
        native_settings.max_substep_delta_time = control_substimestep
        self.carla_world.apply_settings(native_settings)
        self._control_timestep = control_timestep
        self._control_subtimestep = control_substimestep

    def __on_tick_recv(self, world_snapshot : carla.WorldSnapshot):
        self._last_tick_time = world_snapshot.timestamp.elapsed_seconds
    
    @roar_py_thread_sync
    async def step(self) -> float:
        if self.is_asynchronous:
            start_time = self._last_tick_time
            while self._last_tick_time == start_time:
                await asyncio.sleep(__class__.ASYNC_SLEEP_TIME)
                # Instead of waitForTick, we use sleep here to avoid blocking the event loop
            
            if start_time is None:
                dt = 0.0
            else:
                dt = self._last_tick_time - start_time
            return dt
        else:
            self.carla_world.tick(seconds=60.0) # server waits 60s for client to finish the tick
            # self._last_tick_time = self.carla_world.get_snapshot().timestamp.elapsed_seconds # get the timestamp of the last tick
            self._last_tick_time += self.control_timestep
            return self.control_timestep
    
    @property
    def last_tick_elapsed_seconds(self) -> float:
        return self._last_tick_time

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
        # Check if there exists any overriding asset
        spawn_point_asset_dir = __class__.ASSET_DIR + "/spawn_points"
        spawn_point_file = spawn_point_asset_dir + "/" + self.map_name + ".npz"
        if os.path.exists(spawn_point_file):
            spawn_points = np.load(spawn_point_file)
            ret = []
            for i in range(len(spawn_points["locations"])):
                location = spawn_points["locations"][i]
                rotation = spawn_points["rotations"][i]
                ret.append((location, rotation))
            return ret

        native_spawn_points = self._native_carla_map.get_spawn_points()
        ret = []
        for native_spawn_point in native_spawn_points:
            transform_spawn = transform_from_carla(native_spawn_point)
            ret.append(transform_spawn)
        return ret

    """
    Convert a location in world coordinate to geolocation
    Output: [latitude (deg), longitude (deg), altitude(m)]
    """
    def get_geolocation(self, location : np.ndarray) -> np.ndarray:
        assert location.shape == (3,)
        native_location = location_to_carla(location)
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
        roll_pitch_yaw: np.ndarray,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
        bind_to: typing.Optional[carla.Actor] = None
    ) -> carla.Actor:
        assert location.shape == (3,) and roll_pitch_yaw.shape == (3,)
        transform = transform_to_carla(location, roll_pitch_yaw)
        new_actor = self.carla_world.try_spawn_actor(blueprint, transform, bind_to, attachment_type)
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

    @roar_py_append_item
    @roar_py_thread_sync
    def attach_camera_sensor(
        self,
        target_datatype: typing.Type[RoarPyCameraSensorData],
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray,
        fov: float = 90.0,
        image_width: int = 800,
        image_height: int = 600,
        control_timestep: float = 0.0,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
        name: str = "carla_camera",
        bind_to: typing.Optional[RoarPyCarlaActor] = None
    ) -> typing.Optional[RoarPyCameraSensor]:
        
        if target_datatype not in RoarPyCarlaCameraSensor.SUPPORTED_TARGET_DATA_TO_BLUEPRINT:
            raise ValueError(f"Unsupported target data type {target_datatype}")

        blueprint_id = RoarPyCarlaCameraSensor.SUPPORTED_TARGET_DATA_TO_BLUEPRINT[target_datatype]
        blueprint = self.find_blueprint(blueprint_id)
        blueprint.set_attribute("image_size_x", str(image_width))
        blueprint.set_attribute("image_size_y", str(image_height))
        blueprint.set_attribute("fov", str(fov))
        blueprint.set_attribute("sensor_tick", str(control_timestep))
        new_actor = self._attach_native_carla_actor(blueprint, location, roll_pitch_yaw, attachment_type, bind_to._base_actor if bind_to is not None else None)
        
        if new_actor is None:
            return None
        
        new_sensor = RoarPyCarlaCameraSensor(self.carla_instance, new_actor, target_datatype, name=name)
        
        if bind_to is not None:
            bind_to._internal_sensors.append(new_sensor)
        else:
            self._sensors.append(new_sensor)
        return new_sensor
    
    @roar_py_append_item
    @roar_py_thread_sync
    def attach_lidar_sensor(
        self,
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray,
        num_lasers: int = 32,
        max_distance: float = 10.0,
        points_per_second: int = 56000,
        rotation_frequency: float = 10.0,
        upper_fov: float = 10.0,
        lower_fov: float = -30.0,
        horizontal_fov: float = 360.0,
        atmosphere_attenuation_rate: float = 0.004,
        dropoff_general_rate: float = 0.45,
        dropoff_intensity_limit_below: float = 0.8,
        control_timestep: float = 0.0,
        noise_std: float = 0.0,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
        name: str = "carla_lidar_sensor",
        bind_to: typing.Optional[RoarPyCarlaActor] = None
    ) -> typing.Optional[RoarPyLiDARSensor]:
        blueprint = self.find_blueprint("sensor.lidar.ray_cast")
        blueprint.set_attribute("channels", str(num_lasers))
        blueprint.set_attribute("range", str(max_distance))
        blueprint.set_attribute("points_per_second", str(points_per_second))
        blueprint.set_attribute("rotation_frequency", str(rotation_frequency))
        blueprint.set_attribute("upper_fov", str(upper_fov))
        blueprint.set_attribute("lower_fov", str(lower_fov))
        blueprint.set_attribute("horizontal_fov", str(horizontal_fov))
        blueprint.set_attribute("atmosphere_attenuation_rate", str(atmosphere_attenuation_rate))
        blueprint.set_attribute("dropoff_general_rate", str(dropoff_general_rate))
        blueprint.set_attribute("dropoff_intensity_limit", str(dropoff_intensity_limit_below))
        blueprint.set_attribute("sensor_tick", str(control_timestep))
        blueprint.set_attribute("noise_stddev", str(noise_std))

        new_actor = self._attach_native_carla_actor(blueprint, location, roll_pitch_yaw, attachment_type, bind_to._base_actor if bind_to is not None else None)

        if new_actor is None:
            return None

        new_sensor = RoarPyCarlaLiDARSensor(self.carla_instance, new_actor, name=name)

        if bind_to is not None:
            bind_to._internal_sensors.append(new_sensor)
        else:
            self._sensors.append(new_sensor)
        return new_sensor

    @roar_py_remove_item
    @roar_py_thread_sync
    def remove_actor(self, actor : RoarPyActor):
        self._actors.remove(actor)
        actor.close()
    
    @roar_py_remove_item
    @roar_py_thread_sync
    def remove_sensor(self, sensor : RoarPySensor):
        found_sensor = False
        if sensor in self._sensors:
            self._sensors.remove(sensor)
            found_sensor = True
        else:
            for actor in self._actors:
                if sensor in actor._internal_sensors:
                    actor._internal_sensors.remove(sensor)
                    found_sensor = True
                    break
        if not found_sensor:
            raise RuntimeError(f"Sensor {sensor} not found in the world")
        sensor.close()
            
        