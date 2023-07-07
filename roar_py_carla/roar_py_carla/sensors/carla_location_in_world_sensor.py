from roar_py_interface import RoarPyLocationInWorldSensor, RoarPyLocationInWorldSensorData, roar_py_thread_sync
from ..base import RoarPyCarlaBase
import typing

class RoarPyCarlaLocationInWorldSensor(RoarPyLocationInWorldSensor):
    def __init__(
            self, 
            carla_instance: "RoarPyCarlaInstance", 
            binded_target: RoarPyCarlaBase, 
            name: str = "carla_location_in_world_sensor",
    ):
        super().__init__(name, control_timestep=0.0)
        self._carla_instance = carla_instance
        self.binded_target = binded_target
        self.received_data : typing.Optional[RoarPyLocationInWorldSensorData] = None
        self._closed = False

    async def receive_observation(self) -> RoarPyLocationInWorldSensorData:
        self.received_data = RoarPyLocationInWorldSensorData(
            *self.binded_target.get_3d_location()
        )
        return self.received_data
    
    def get_last_observation(self) -> typing.Optional[RoarPyLocationInWorldSensorData]:
        return self.received_data
    
    @roar_py_thread_sync
    def close(self):
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed
