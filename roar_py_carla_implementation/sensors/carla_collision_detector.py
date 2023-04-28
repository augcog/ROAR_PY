from roar_py_interface import RoarPyCollisionSensor, RoarPyCollisionSensorData
from dataclasses import dataclass
import typing
import asyncio
import numpy as np
import gymnasium as gym
import carla

class RoarPyCollisionSensor(RoarPyCollisionSensor[RoarPyCollisionSensorData]):
    def __init__(
        self, 
        sensor: carla.Sensor,
        name: str = "carla_collision_sensor",
    ):
        assert sensor.type_id == "sensor.other.collision", "Unsupported blueprint_id: {} for carla collision sensor support".format(sensor.type_id)
        super().__init__(name, control_timestep = 0.0)
        self.received_data : typing.Optional[RoarPyCollisionSensorData] = None
        self.sensor = sensor
        self.sensor.listen(
            self.listen_callback
        )

    async def receive_observation(self) -> RoarPyCollisionSensorData:
        while self.received_data is None:
            await asyncio.sleep(0.001)
        return self.received_data
    
    def listen_callback(self, event: carla.CollisionEvent):
        self.received_data = RoarPyCollisionSensorData(np.array([
            event.actor,
            event.other_actor,
            [
                event.normal_impulse.x, 
                event.normal_impulse.y, 
                event.normal_impulse.z
            ]
        ]))
    
    def get_last_observation(self) -> typing.Optional[RoarPyCollisionSensorData]:
        return self.received_data
    
    def close(self):
        if self.sensor is not None and self.sensor.is_listening:
            self.sensor.stop()
            self.sensor = None
    
    def is_closed(self) -> bool:
        return self.sensor is None or not self.sensor.is_listening
    
    @property
    def other_actor(self) -> carla.Actor | None:
        if self.received_data is None:
            return None
        else:
            return self.received_data.other_actor
        
    @property
    def impulse_normal(self) -> np.array:
        return self.received_data.impulse_normal
