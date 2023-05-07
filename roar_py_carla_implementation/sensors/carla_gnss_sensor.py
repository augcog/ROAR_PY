from roar_py_interface import RoarPyGNSSSensor, RoarPyGNSSSensorData
from dataclasses import dataclass
import typing
import asyncio
import numpy as np
import gymnasium as gym
import carla
from ..base import RoarPyCarlaBase
from ..clients import RoarPyCarlaInstance

class RoarPyCarlaGNSSSensor(RoarPyGNSSSensor[RoarPyGNSSSensorData], RoarPyCarlaBase):
    def __init__(
        self, 
        carla_instance: RoarPyCarlaInstance,
        sensor: carla.Sensor,
        name: str = "carla_gnss_sensor",
    ):
        assert sensor.type_id == "sensor.other.gnss", "Unsupported blueprint_id: {} for carla gnss sensor support".format(sensor.type_id)
        RoarPyCarlaGNSSSensor.__init__(self, name, control_timestep = 0.0)
        RoarPyCarlaBase.__init__(self, carla_instance, sensor)
        self.received_data : typing.Optional[RoarPyGNSSSensorData] = None
        sensor.listen(
            self.listen_callback
        )

    async def receive_observation(self) -> RoarPyGNSSSensorData:
        while self.received_data is None:
            await asyncio.sleep(0.001)
        return self.received_data
    
    def listen_carla_data(self, gnss_data: carla.GnssMeasurement) -> None:
        self.received_data = RoarPyGNSSSensorData(
            gnss_data.altitude,
            gnss_data.latitude,
            gnss_data.longitude
        )

    def get_last_observation(self) -> typing.Optional[RoarPyGNSSSensorData]:
        return self.received_data
    
    def close(self):
        if self._base_actor is not None and self._base_actor.is_listening:
            self._base_actor.stop()
        RoarPyCarlaBase.close(self)

    def is_closed(self) -> bool:
        return self._base_actor is None or not self._base_actor.is_listening
