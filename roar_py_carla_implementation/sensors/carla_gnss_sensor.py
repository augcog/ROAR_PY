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
        RoarPyCarlaBase.__init__(self, carla_instance, sensor)
        RoarPyCarlaGNSSSensor.__init__(self, name, control_timestep = 0.0)
        self.received_data : typing.Optional[RoarPyGNSSSensorData] = None
        sensor.listen(
            self.listen_callback
        )
    
    @property
    def noise_altitude_bias(self) -> float:
        return self._base_actor.noise_alt_bias

    @noise_altitude_bias.setter
    def noise_altitude_bias(self, value: float) -> None:
        self._base_actor.noise_alt_bias = value

    @property
    def noise_altitude_std(self) -> float:
        return self._base_actor.noise_alt_stddev

    @noise_altitude_std.setter
    def noise_altitude_std(self, value: float) -> None:
        self._base_actor.noise_alt_stddev = value
    
    @property
    def noise_latitude_bias(self) -> float:
        return self._base_actor.noise_lat_bias
    
    @noise_latitude_bias.setter
    def noise_latitude_bias(self, value: float) -> None:
        self._base_actor.noise_lat_bias = value
    
    @property
    def noise_latitude_std(self) -> float:
        return self._base_actor.noise_lat_stddev
    
    @noise_latitude_std.setter
    def noise_latitude_std(self, value: float) -> None:
        self._base_actor.noise_lat_stddev = value
    
    @property
    def noise_longitude_bias(self) -> float:
        return self._base_actor.noise_lon_bias
    
    @noise_longitude_bias.setter
    def noise_longitude_bias(self, value: float) -> None:
        self._base_actor.noise_lon_bias = value

    @property
    def noise_longitude_std(self) -> float:
        return self._base_actor.noise_lon_stddev
    
    @noise_longitude_std.setter
    def noise_longitude_std(self, value: float) -> None:
        self._base_actor.noise_lon_stddev = value

    @property
    def noise_seed(self) -> int:
        return self._base_actor.noise_seed
    
    @noise_seed.setter
    def noise_seed(self, value: int) -> None:
        self._base_actor.noise_seed = value

    @property
    def control_timestep(self) -> float:
        return self._base_actor.sensor_tick
    
    @control_timestep.setter
    def control_timestep(self, value: float) -> None:
        self._base_actor.sensor_tick = value

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
