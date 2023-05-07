import carla
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
from ..clients import RoarPyCarlaInstance
from ..worlds import RoarPyCarlaWorld

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
        carla_instance : RoarPyCarlaInstance,
        base_actor : carla.Actor,
    ) -> None:
        self._base_actor = base_actor
        self._carla_instance = carla_instance
        carla_instance.register_actor(base_actor.id, self)
    
    @property
    def carla_attributes(self) -> Dict:
        return self._base_actor.attributes
    
    @property
    def carla_id(self) -> int:
        return self._base_actor.id
    
    @property
    def carla_blueprint_type_id(self) -> str:
        return self._base_actor.type_id
    
    @property
    def carla_is_alive(self) -> bool:
        return self._base_actor.is_alive
    
    @property
    def carla_parent(self) -> Optional[carla.Actor]:
        return self._base_actor.parent
    
    def get_native_carla_world(self) -> carla.World:
        return self._base_actor.get_world()
    
    def get_carla_world(self) -> RoarPyCarlaWorld:
        return self._carla_instance.world

    @property
    def semantic_labels(self) -> List[int]:
        # Return the semantic labels of the actor, see carla_camera_rgb.py for example tags
        return self._base_actor.semantic_tags
    
    @property
    def bounding_box(self) -> RoarPyCarlaBoundingBox:
        bdBox = self._base_actor.bounding_box
        return RoarPyCarlaBoundingBox(
            extent=np.array([bdBox.extent.x, bdBox.extent.y, bdBox.extent.z]),
            location=np.array([bdBox.location.x, bdBox.location.y, bdBox.location.z]),
            rotation=np.deg2rad(np.array([bdBox.rotation.roll, bdBox.rotation.pitch, bdBox.rotation.yaw]))
        )

    def get_acceleration(self) -> np.ndarray:
        acc = self._base_actor.get_acceleration()
        return np.array([acc.x, acc.y, acc.z])
    
    # Angular velocity in radians per second
    def get_angular_velocity(self) -> np.ndarray:
        ang_vel = self._base_actor.get_angular_velocity()
        return np.deg2rad(np.array([ang_vel.x, ang_vel.y, ang_vel.z]))
    
    # Angular velocity in radians per second
    def set_angular_velocity(self, target_angular_velocity : np.ndarray):
        ang_vel = np.rad2deg(target_angular_velocity)
        self._base_actor.set_target_angular_velocity(carla.Vector3D(x=ang_vel[0], y=ang_vel[1], z=ang_vel[2]))
    
    # Linear velocity in meters per second (in world frame)
    def get_linear_3d_velocity(self) -> np.ndarray:
        vel = self._base_actor.get_velocity()
        return np.array([vel.x, vel.y, vel.z])
    
    def set_linear_3d_velocity(self, target_linear_velocity : np.ndarray) -> None:
        self._base_actor.set_target_velocity(carla.Vector3D(x=target_linear_velocity[0], y=target_linear_velocity[1], z=target_linear_velocity[2]))
    
    def get_3d_location(self) -> np.ndarray:
        loc = self._base_actor.get_location()
        return np.array([loc.x, loc.y, loc.z])
    
    def set_3d_location(self, new_location: np.ndarray) -> None:
        self._base_actor.set_location(carla.Location(x=new_location[0], y=new_location[1], z=new_location[2]))
    
    # Get the rotation of the actor in radians
    def get_roll_pitch_yaw(self) -> np.ndarray:
        rot = self._base_actor.get_transform().rotation
        return np.deg2rad(np.array([rot.roll, rot.pitch, rot.yaw]))
    
    def set_roll_pitch_yaw(self, new_rotation_rpy : np.ndarray) -> None:
        new_rot_deg = np.rad2deg(new_rotation_rpy)
        transform = carla.Transform(
            location=self._base_actor.get_location(),
            rotation=carla.Rotation(roll=new_rot_deg[0], pitch=new_rot_deg[1], yaw=new_rot_deg[2])
        )
        self._base_actor.set_transform(transform)
    
    def set_transform(self, new_location : np.ndarray, new_rotation : np.ndarray) -> None:
        new_rot_deg = np.rad2deg(new_rotation)
        transform = carla.Transform(
            location=carla.Location(x=new_location[0], y=new_location[1], z=new_location[2]),
            rotation=carla.Rotation(roll=new_rot_deg[0], pitch=new_rot_deg[1], yaw=new_rot_deg[2])
        )
        self._base_actor.set_transform(transform)

    
    def set_enable_gravity(self, enable: bool = True) -> None:
        self._base_actor.set_enable_gravity(enable)

    def set_simulate_physics(self, enable: bool = True) -> None:
        self._base_actor.set_simulate_physics(enable)
    
    def attach_native_carla_actor(
        self,
        blueprint_id : str, 
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray,
        attachment_type: carla.AttachmentType = carla.AttachmentType.Rigid
    ) -> carla.Actor:
        assert location.shape == (3,) and roll_pitch_yaw.shape == (3,)
        blueprint = self.get_native_carla_world().get_blueprint_library().find(blueprint_id)
        transform = carla.Transform(carla.Location(*location), carla.Rotation(roll=roll_pitch_yaw[0], pitch=roll_pitch_yaw[1], yaw=roll_pitch_yaw[2]))
        new_actor = self.get_native_carla_world().spawn_actor(blueprint, transform, attach_to=self._base_actor, attachment=attachment_type)
        return new_actor

    def __str__(self):
        return str(self._base_actor)
    
    def close(self):
        if self._base_actor.is_alive:
            self._base_actor.destroy()
        self._carla_instance.unregister_actor(self._base_actor.id, self)

    def is_closed(self) -> bool:
        return self._base_actor.is_alive