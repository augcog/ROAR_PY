from roar_py_interface import RoarPyGyroscopeSensor, RoarPyGyroscopeSensorData, roar_py_thread_sync
import numpy as np
import typing
from ..base import RoarPyCarlaBase

class RoarPyCarlaGyroscopeSensor(RoarPyGyroscopeSensor):
    def __init__(self, carla_instance : "RoarPyCarlaInstance", bind_target : RoarPyCarlaBase, name: str = "carla_gyroscope_sensor"):
        super().__init__(name, control_timestep=0.0)
        self._carla_instance = carla_instance
        self.bind_target = bind_target
        self.received_data : typing.Optional[RoarPyGyroscopeSensorData] = None
        self._closed = False
    
    async def receive_observation(self) -> RoarPyGyroscopeSensorData:
        self.received_data = RoarPyGyroscopeSensorData(
            self.bind_target.get_angular_velocity()
        )
        return self.received_data
    
    def get_last_observation(self) -> typing.Optional[RoarPyGyroscopeSensorData]:
        return self.received_data
    
    @roar_py_thread_sync
    def close(self):
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed