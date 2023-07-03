from typing_extensions import override
from roar_py_interface.actors.actor import RoarPyActor, RoarPyResettableActor
from roar_py_interface.sensors import *
from roar_py_interface.base import RoarPySensor
from roar_py_interface.wrappers import roar_py_thread_sync, roar_py_append_item, roar_py_remove_item
from roar_py_interface.worlds.occupancy_map import RoarPyOccupancyMapProducer
import typing
import gymnasium as gym
import carla
import transforms3d as tr3d
from ..base import RoarPyCarlaBase
from ..sensors import *
import numpy as np

class RoarPyCarlaActor(RoarPyActor, RoarPyCarlaBase):
    def __init__(
        self, 
        carla_instance: "RoarPyCarlaInstance",
        carla_actor: carla.Actor,
        name: str
    ):
        RoarPyActor.__init__(self, control_timestep=0.0, force_real_control_timestep=False, name=name)
        RoarPyCarlaBase.__init__(self, carla_instance, carla_actor)
        self._internal_sensors : typing.List[RoarPySensor] = []

    def get_action_spec(self) -> gym.Space:
        raise NotImplementedError()
    
    @override
    async def _apply_action(self, action: typing.Any) -> bool:
        raise NotImplementedError()
    
    def _refresh_sensor_list(self):
        new_sensor_list = []
        for sensor in self._internal_sensors:
            if not sensor.is_closed():
                new_sensor_list.append(sensor)
        self._internal_sensors = new_sensor_list

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        self._refresh_sensor_list()
        return self._internal_sensors.copy()

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
        name: str = "carla_camera"
    ) -> typing.Optional[RoarPyCameraSensor]:
        return self._get_carla_world().attach_camera_sensor(
            target_datatype,
            location,
            roll_pitch_yaw,
            fov,
            image_width,
            image_height,
            control_timestep,
            attachment_type,
            name,
            self
        )

    @roar_py_append_item
    @roar_py_thread_sync
    def attach_collision_sensor(
        self,
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
        name: str = "carla_collision_sensor",
    ) -> typing.Optional[RoarPyCollisionSensor]:
        blueprint = self._get_carla_world().find_blueprint("sensor.other.collision")
        new_actor = self._attach_native_carla_actor(blueprint, location, roll_pitch_yaw, attachment_type)

        if new_actor is None:
            return None

        new_sensor = RoarPyCarlaCollisionSensor(self._carla_instance, new_actor, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor

    @roar_py_append_item
    @roar_py_thread_sync
    def attach_accelerometer_sensor(
        self,
        name : str = "carla_accelerometer_sensor",
    ):
        new_sensor = RoarPyCarlaAccelerometerSensor(self._carla_instance, self, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor
    
    @roar_py_append_item
    @roar_py_thread_sync
    def attach_velocimeter_sensor(
        self,
        name : str = "carla_velocimeter_sensor",
    ):
        new_sensor = RoarPyCarlaVelocimeterSensor(self._carla_instance, self, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor
    
    @roar_py_append_item
    @roar_py_thread_sync
    def attach_local_velocimeter_sensor(
        self,
        name : str = "carla_local_velocimeter_sensor",
    ):
        new_sensor = RoarPyCarlaLocalVelocimeterSensor(self._carla_instance, self, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor

    @roar_py_append_item
    @roar_py_thread_sync
    def attach_gnss_sensor(
        self,
        noise_altitude_bias: float = 0.0,
        noise_altitude_std: float = 0.0,
        noise_latitude_bias: float = 0.0,
        noise_latitude_std: float = 0.0,
        noise_longitude_bias: float = 0.0,
        noise_longitude_std: float = 0.0,
        noise_seed: int = 0,
        control_timestep: float = 0.0,
        name : str = "carla_gnss_sensor",
    ) -> typing.Optional[RoarPyGNSSSensor]:
        blueprint = self._get_carla_world().find_blueprint("sensor.other.gnss")
        blueprint.set_attribute("noise_alt_bias", str(noise_altitude_bias))
        blueprint.set_attribute("noise_alt_stddev", str(noise_altitude_std))
        blueprint.set_attribute("noise_lat_bias", str(noise_latitude_bias))
        blueprint.set_attribute("noise_lat_stddev", str(noise_latitude_std))
        blueprint.set_attribute("noise_lon_bias", str(noise_longitude_bias))
        blueprint.set_attribute("noise_lon_stddev", str(noise_longitude_std))
        blueprint.set_attribute("noise_seed", str(noise_seed))
        blueprint.set_attribute("sensor_tick", str(control_timestep))
        new_actor = self._attach_native_carla_actor(blueprint, np.array([0,0,0]), np.array([0,0,0]), carla.AttachmentType.Rigid)

        if new_actor is None:
            return None

        new_sensor = RoarPyCarlaGNSSSensor(self._carla_instance, new_actor, name=name)
        self._internal_sensors.append(new_sensor)
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
        name: str = "carla_lidar_sensor"
    ) -> typing.Optional[RoarPyLiDARSensor]:
        return self._get_carla_world().attach_lidar_sensor(
            location,
            roll_pitch_yaw,
            num_lasers,
            max_distance,
            points_per_second,
            rotation_frequency,
            upper_fov,
            lower_fov,
            horizontal_fov,
            atmosphere_attenuation_rate,
            dropoff_general_rate,
            dropoff_intensity_limit_below,
            control_timestep,
            noise_std,
            attachment_type,
            name,
            self
        )

    @roar_py_append_item
    @roar_py_thread_sync
    def attach_roll_pitch_yaw_sensor(
        self,
        name: str = "carla_rpy_sensor",
    ):
        new_sensor = RoarPyCarlaRPYSensor(self._carla_instance, self, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor
    
    @roar_py_append_item
    @roar_py_thread_sync
    def attach_framequat_sensor(
        self,
        name: str = "carla_framequat_sensor",
    ):
        new_sensor = RoarPyCarlaRPYSensor(self._carla_instance, self)
        new_sensor = RoarPyFrameQuatSensorFromRollPitchYaw(new_sensor, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor
    
    @roar_py_append_item
    @roar_py_thread_sync
    def attach_gyroscope_sensor(
        self,
        name: str = "carla_gyroscope_sensor",
    ):
        new_sensor = RoarPyCarlaGyroscopeSensor(self._carla_instance, self, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor
    
    @roar_py_append_item
    @roar_py_thread_sync
    def attach_location_in_world_sensor(
        self,
        name: str = "carla_location_in_world_sensor",
    ):
        new_sensor = RoarPyCarlaLocationInWorldSensor(self._carla_instance, self, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor
    
    @roar_py_append_item
    @roar_py_thread_sync
    def attach_occupancy_map_sensor(
        self,
        width : int,
        height : int,
        width_in_world : float,
        height_in_world : float,
        name: str = "carla_occupancy_map_sensor",
    ):
        world = self._get_carla_world()
        new_sensor = RoarPyCarlaOccupancyMapSensor(
            RoarPyOccupancyMapProducer(
                world.maneuverable_waypoints,
                width,
                height,
                width_in_world,
                height_in_world,
            ),
            self,
            name
        )
        self._internal_sensors.append(new_sensor)
        return new_sensor

    @roar_py_remove_item
    @roar_py_thread_sync
    def remove_sensor(self, sensor: RoarPySensor):
        self._internal_sensors.remove(sensor)
        if not sensor.is_closed():
            sensor.close()
    
    @roar_py_thread_sync
    def close(self):
        for sensor in self._internal_sensors:
            if not sensor.is_closed():
                sensor.close()
        self._internal_sensors = []
        RoarPyCarlaBase.close(self)

    @roar_py_thread_sync
    def is_closed(self) -> bool:
        return RoarPyCarlaBase.is_closed(self) and len(self._internal_sensors) == 0