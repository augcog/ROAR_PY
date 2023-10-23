from roar_py_interface import RoarPyCollisionSensor, RoarPyCollisionSensorData, roar_py_thread_sync
from dataclasses import dataclass
import typing
import asyncio
import numpy as np
import gymnasium as gym
import carla
from ..base import RoarPyCarlaBase

class RoarPyCarlaCollisionSensor(RoarPyCollisionSensor, RoarPyCarlaBase):
    def __init__(
        self, 
        carla_instance: "RoarPyCarlaInstance",
        sensor: carla.Sensor,
        name: str = "carla_collision_sensor",
    ):
        assert sensor.type_id == "sensor.other.collision", "Unsupported blueprint_id: {} for carla collision sensor support".format(sensor.type_id)
        RoarPyCollisionSensor.__init__(self, name, control_timestep = 0.0)
        RoarPyCarlaBase.__init__(self, carla_instance, sensor)
        self.received_data : RoarPyCollisionSensorData = None
        self.new_data = RoarPyCollisionSensorData(
            None,
            None,
            None
        )
        sensor.listen(
            self.listen_callback
        )

    async def receive_observation(self) -> RoarPyCollisionSensorData:
        ret = self.new_data
        self.received_data = self.new_data
        self.new_data = RoarPyCollisionSensorData(
            None,
            None,
            None
        )
        return ret
    
    def listen_callback(self, event: carla.CollisionEvent):
        current_actor = self._carla_instance.search_actor(event.actor.id)
        other_actor = self._carla_instance.search_actor(event.other_actor.id)
        impulse_normal = np.array([
            event.normal_impulse.x, 
            event.normal_impulse.y, 
            event.normal_impulse.z
        ])

        self.new_data.actor = current_actor

        if self.new_data.other_actor is None:
            self.new_data.other_actor = [other_actor]
        else:
            self.new_data.other_actor.append(other_actor)
        
        if self.new_data.impulse_normals is None:
            self.new_data.impulse_normals = [impulse_normal]
        else:
            self.new_data.impulse_normals.append(impulse_normal)
    
    def get_last_observation(self) -> typing.Optional[RoarPyCollisionSensorData]:
        return self.received_data
    
    @roar_py_thread_sync
    def close(self):
        if self._base_actor is not None and self._base_actor.is_listening:
            self._base_actor.stop()
        RoarPyCarlaBase.close(self)
    
    @roar_py_thread_sync
    def is_closed(self) -> bool:
        return self._base_actor is None or not self._base_actor.is_listening