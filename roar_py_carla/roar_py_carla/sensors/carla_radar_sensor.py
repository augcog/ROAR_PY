from roar_py_core.roar_py_interface import RoarPyRadarSensor, RoarPyRadarSensorData, roar_py_thread_sync
import typing
import gymnasium as gym
import carla
import asyncio
import numpy as np
from PIL import Image
from ..base import RoarPyCarlaBase

def __convert_carla_radar_raw_to_roar_py(carla_radar_dat : carla.RadarMeasurement) -> RoarPyRadarSensorData:
    p_cloud_size = len(carla_radar_dat)
    p_cloud = np.copy(np.frombuffer(carla_radar_dat.raw_data, dtype=np.dtype('f4')))
    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
    return RoarPyRadarSensorData(p_cloud)

class RoarPyCarlaRadarSensor(RoarPyRadarSensor, RoarPyCarlaBase):
    def __init__(
        self, 
        carla_instance: "RoarPyCarlaInstance",
        sensor: carla.Sensor,
        target_data_type: typing.Optional[typing.Type[RoarPyRadarSensorData]] = None,
        name: str = "carla_radar_sensor",
    ):
        assert sensor.type_id == "sensor.other.radar", "Unsupported blueprint_id: {} for carla collision sensor support".format(sensor.type_id)
        RoarPyRadarSensor.__init__(self, name, control_timestep = 0.0)
        RoarPyCarlaBase.__init__(self, carla_instance, sensor)
        self.received_data : typing.Optional[RoarPyRadarSensorData] = None
        sensor.listen(
            self.listen_carla_data
        )

    @property
    def control_timestep(self) -> float:
        return float(self._base_actor.attributes["sensor_tick"])
    
    # In radians
    @property
    def horizontal_fov(self) -> float:
        return float(self._base_actor.attributes["horizontal_fov"])
    
    @property
    def points_per_second(self) -> float:
        return float(self._base_actor.attributes["points_per_second"])
    
    # In meters
    @property
    def max_distance(self) -> float:
        return float(self._base_actor.attributes["range"])
    
    # In radians
    @property
    def vertical_fov(self) -> float:
        return float(self._base_actor.attributes["vertical_fov"])
    
    async def receive_observation(self) -> RoarPyRadarSensorData:
        while self.received_data is None:
            await asyncio.sleep(0.001)
        return self.received_data
    
    def listen_carla_data(self, carla_data: carla.RadarMeasurement) -> None:
        self.received_data = __convert_carla_radar_raw_to_roar_py(carla_data)
    
    def get_last_observation(self) -> typing.Optional[RoarPyRadarSensorData]:
        return self.received_data
    
    @roar_py_thread_sync
    def close(self):
        if self._base_actor is not None and self._base_actor.is_listening:
            self._base_actor.stop()
        RoarPyCarlaBase.close(self)

    @roar_py_thread_sync
    def is_closed(self) -> bool:
        return self._base_actor is None or not self._base_actor.is_listening