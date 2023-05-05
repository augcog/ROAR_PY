from roar_py_interface import RoarPyRollPitchYawSensor, RoarPyRollPitchYawSensorData
from dataclasses import dataclass
import typing
import asyncio
import numpy as np
import gymnasium as gym
import carla
from ..base import RoarPyCarlaBase

class RoarPyCarlaRPYSensor(RoarPyRollPitchYawSensor[RoarPyRollPitchYawSensorData]):
    def __init__(self, binded_target : RoarPyCarlaBase, name: str = "rpy_sensor"):
        super().__init__(name, control_timestep = 0.0)
        self.binded_target = binded_target
        self.received_data : typing.Optional[RoarPyRollPitchYawSensorData] = None
        self._closed = False
    
    async def receive_observation(self) -> RoarPyRollPitchYawSensorData:
        rotation = self.binded_target.get_roll_pitch_yaw()
        self.received_data = RoarPyRollPitchYawSensorData(
            rotation
        )
        return self.received_data
    
    def get_last_observation(self) -> typing.Optional[RoarPyRollPitchYawSensorData]:
        return self.received_data
    
    def close(self):
        self._closed = True
    
    def is_closed(self) -> bool:
        return self._closed