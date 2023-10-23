import carla
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
from roar_py_interface.wrappers import roar_py_thread_sync
from ..utils import *

@dataclass
class RoarPyCarlaBoundingBox:
    """
    Vector from the center of the box to one vertex. In meters
    The value in each axis equals half the size of the box for that axis. extent.x * 2 would return the size of the box in the X-axis.
    """
    extent : np.ndarray
    
    """
    The center of the bounding box. In meters
    """
    location : np.ndarray

    """
    roll, pitch, yaw in radians of the bounding box
    """
    rotation : np.ndarray

class RoarPyCarlaBase:
    def __init__(
        self,
        carla_instance : "RoarPyCarlaInstance",
        base_actor : carla.Actor,
    ) -> None:
        self._base_actor = base_actor
        self._carla_instance = carla_instance
        carla_instance.register_actor(base_actor.id, self)
    
    @property
    def parent(self) -> Optional["RoarPyCarlaBase"]:
        if self._base_actor.parent is None:
            return None
        else:
            return self._carla_instance.search_actor(self._base_actor.parent.id)

    @property
    @roar_py_thread_sync
    def carla_attributes(self) -> Dict:
        return self._base_actor.attributes
    
    @property
    @roar_py_thread_sync
    def carla_id(self) -> int:
        return self._base_actor.id
    
    @property
    @roar_py_thread_sync
    def carla_blueprint_type_id(self) -> str:
        return self._base_actor.type_id
    
    @property
    @roar_py_thread_sync
    def carla_is_alive(self) -> bool:
        return self._base_actor.is_alive
    
    def _get_native_carla_world(self) -> carla.World:
        return self._base_actor.get_world()
    
    def _get_carla_world(self):
        return self._carla_instance.world

    @property
    def semantic_labels(self) -> List[int]:
        # Return the semantic labels of the actor, see carla_camera_rgb.py for example tags
        return self._base_actor.semantic_tags
    
    @property
    def bounding_box(self) -> RoarPyCarlaBoundingBox:
        bdBox = self._base_actor.bounding_box
        
        return RoarPyCarlaBoundingBox(
            extent=np.abs(location_from_carla(bdBox.extent)),#np.array([bdBox.extent.x, bdBox.extent.y, bdBox.extent.z]),
            location=location_from_carla(bdBox.location),
            rotation=rotation_from_carla(bdBox.rotation)
        )

    def get_acceleration(self) -> np.ndarray:
        acc = self._base_actor.get_acceleration()
        return location_from_carla(carla.Location(x=acc.x, y=acc.y, z=acc.z))
    
    # Angular velocity in radians per second
    def get_angular_velocity(self) -> np.ndarray:
        ang_vel = self._base_actor.get_angular_velocity()
        return rotation_from_carla(carla.Rotation(roll=ang_vel.x, pitch=ang_vel.y, yaw=ang_vel.z))
    
    # Angular velocity in radians per second
    @roar_py_thread_sync
    def set_angular_velocity(self, target_angular_velocity : np.ndarray):
        ang_vel = rotation_to_carla(target_angular_velocity)
        self._base_actor.set_target_angular_velocity(carla.Vector3D(x=ang_vel.roll, y=ang_vel.pitch, z=ang_vel.yaw))
    
    # Linear velocity in meters per second (in world frame)
    def get_linear_3d_velocity(self) -> np.ndarray:
        vel = self._base_actor.get_velocity()
        return location_from_carla(vel)
    
    @roar_py_thread_sync
    def set_linear_3d_velocity(self, target_linear_velocity : np.ndarray) -> None:
        target_linear_velocity = location_to_carla(target_linear_velocity)
        self._base_actor.set_target_velocity(carla.Vector3D(x=target_linear_velocity.x, y=target_linear_velocity.y, z=target_linear_velocity.z))
    
    def get_3d_location(self) -> np.ndarray:
        loc = self._base_actor.get_location()
        return location_from_carla(loc)
    
    @roar_py_thread_sync
    def set_3d_location(self, new_location: np.ndarray) -> None:
        new_location = location_to_carla(new_location)
        self._base_actor.set_location(new_location)
    
    # Get the rotation of the actor in radians
    def get_roll_pitch_yaw(self) -> np.ndarray:
        rot = self._base_actor.get_transform().rotation
        return rotation_from_carla(rot)
    
    @roar_py_thread_sync
    def set_roll_pitch_yaw(self, new_rotation_rpy : np.ndarray) -> None:
        transform = carla.Transform(
            location=self._base_actor.get_location(),
            rotation=rotation_to_carla(new_rotation_rpy)
        )
        self._base_actor.set_transform(transform)
    
    @roar_py_thread_sync
    def set_transform(self, new_location : np.ndarray, new_rotation : np.ndarray) -> None:
        transform = transform_to_carla(new_location, new_rotation)
        self._base_actor.set_transform(transform)

    @roar_py_thread_sync
    def set_enable_gravity(self, enable: bool = True) -> None:
        self._base_actor.set_enable_gravity(enable)

    @roar_py_thread_sync
    def set_simulate_physics(self, enable: bool = True) -> None:
        self._base_actor.set_simulate_physics(enable)
    
    @roar_py_thread_sync
    def _attach_native_carla_actor(
        self,
        blueprint : carla.ActorBlueprint, 
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid
    ) -> Optional[carla.Actor]:
        # assert location.shape == (3,) and roll_pitch_yaw.shape == (3,)
        # transform = transform_to_carla(location, roll_pitch_yaw)
        # new_actor = self._get_native_carla_world().try_spawn_actor(blueprint, transform, self._base_actor, attachment_type)
        # return new_actor
        return self._get_carla_world()._attach_native_carla_actor(
            blueprint,
            location,
            roll_pitch_yaw,
            attachment_type,
            self._base_actor
        )

    def __str__(self):
        return str(self._base_actor)
    
    def close(self):
        if self._base_actor.is_alive:
            self._base_actor.destroy()
        self._carla_instance.unregister_actor(self._base_actor.id, self)

    def is_closed(self) -> bool:
        return not self._base_actor.is_alive