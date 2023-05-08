from roar_py_interface.actors.actor import RoarPyActor, RoarPyResettableActor
from roar_py_interface.sensors import *
from roar_py_interface.base import RoarPySensor
from roar_py_interface.wrappers import roar_py_thread_sync, roar_py_append_item, roar_py_remove_item
import typing
import gymnasium as gym
import carla
import transforms3d as tr3d
from ..base import RoarPyCarlaBase
from ..clients import RoarPyCarlaInstance
from ..sensors import *
import numpy as np

class RoarPyCarlaActor(RoarPyActor, RoarPyCarlaBase):
    def __init__(
        self, 
        carla_instance: RoarPyCarlaInstance,
        carla_actor: carla.Actor,
        name: str
    ):
        RoarPyActor.__init__(self, control_timestep=0.0, force_real_control_timestep=False, name=name)
        RoarPyCarlaBase.__init__(self, carla_instance, carla_actor)
        self._internal_sensors = []

    def get_action_spec(self) -> gym.Space:
        raise NotImplementedError()
    
    async def __apply_action(self, action: typing.Any) -> bool:
        raise NotImplementedError()
    
    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        return self._internal_sensors

    @roar_py_append_item
    @roar_py_thread_sync
    def attach_camera_sensor(
        self,
        target_datatype: typing.Type[RoarPyCameraSensorData],
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
        name: str = "carla_camera",
    ) -> RoarPyCameraSensor:
        if target_datatype not in RoarPyCarlaCameraSensor.SUPPORTED_TARGET_DATA_TO_BLUEPRINT:
            raise ValueError(f"Unsupported target data type {target_datatype}")

        blueprint_id = RoarPyCarlaCameraSensor.SUPPORTED_TARGET_DATA_TO_BLUEPRINT[target_datatype]
        new_actor = self._attach_native_carla_actor(blueprint_id, location, roll_pitch_yaw, attachment_type)
        new_sensor = RoarPyCarlaCameraSensor(self._carla_instance, new_actor, target_datatype, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor

    @roar_py_append_item
    @roar_py_thread_sync
    def attach_collision_sensor(
        self,
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
        name: str = "carla_collision_sensor",
    ):
        blueprint_id = "sensor.other.collision"
        new_actor = self._attach_native_carla_actor(blueprint_id, location, roll_pitch_yaw, attachment_type)
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
    def attach_gnss_sensor(
        self,
        name : str = "carla_gnss_sensor",
    ):
        blueprint_id = "sensor.other.gnss"
        new_actor = self._attach_native_carla_actor(blueprint_id, np.array([0,0,0]), np.array([0,0,0]), carla.AttachmentType.Rigid)
        new_sensor = RoarPyCarlaGNSSSensor(self._carla_instance, new_actor, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor

    @roar_py_append_item
    @roar_py_thread_sync
    def attach_lidar_sensor(
        self,
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid,
        name: str = "carla_lidar_sensor",
    ):
        blueprint_id = "sensor.lidar.ray_cast"
        new_actor = self._attach_native_carla_actor(blueprint_id, location, roll_pitch_yaw, attachment_type)
        new_sensor = RoarPyCarlaLiDARSensor(self._carla_instance, new_actor, name=name)
        self._internal_sensors.append(new_sensor)
        return new_sensor
    
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