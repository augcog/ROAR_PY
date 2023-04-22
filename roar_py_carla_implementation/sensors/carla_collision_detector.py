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
        name: str,
        sensor: carla.Sensor,
    ):
        super().__init__(name, control_timestep=0.02)
        self.received_data : typing.Optional[RoarPyCollisionSensorData] = None

        assert sensor.type_id == "sensor.other.collision"

    async def receive_observation(self) -> RoarPyCollisionSensorData:
        while self.received_data is None:
            await asyncio.sleep(0.001)
        return self.received_data
    
    def listen_callback(self, event: carla.CollisionEvent):
        self.received_data = RoarPyCollisionSensorData(np.array([
            event.normal_impulse.x, 
            event.normal_impulse.y, 
            event.normal_impulse.z
        ]))
    
    def get_last_observation(self) -> typing.Optional[RoarPyCollisionSensorData]:
        return self.received_data
    
    def close(self):
        if self.sensor is not None and self.sensor.is_listening:
            self.sensor.stop()
            self.sensor = None
    
    def is_closed(self) -> bool:
        return self.sensor is None or not self.sensor.is_listening
