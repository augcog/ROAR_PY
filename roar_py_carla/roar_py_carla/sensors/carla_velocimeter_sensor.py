from roar_py_interface import RoarPyVelocimeterSensor, RoarPyVelocimeterSensorData, roar_py_thread_sync
from ..base import RoarPyCarlaBase
import typing
import transforms3d as tr3d

class RoarPyCarlaVelocimeterSensor(RoarPyVelocimeterSensor):
    def __init__(self, carla_instance : "RoarPyCarlaInstance", binded_target : RoarPyCarlaBase, name: str = "carla_velocimeter_sensor"):
        super().__init__(name, control_timestep=0.0)
        self._carla_instance = carla_instance
        self.binded_target = binded_target
        self.received_data : typing.Optional[RoarPyVelocimeterSensorData] = None
        self._closed = False
    
    async def receive_observation(self) -> RoarPyVelocimeterSensorData:
        self.received_data = RoarPyVelocimeterSensorData(
            self.binded_target.get_linear_3d_velocity()
        )
        return self.received_data
    
    def get_last_observation(self) -> typing.Optional[RoarPyVelocimeterSensorData]:
        return self.received_data
    
    @roar_py_thread_sync
    def close(self):
        self._closed = True
    
    def is_closed(self) -> bool:
        return self._closed

class RoarPyCarlaLocalVelocimeterSensor(RoarPyCarlaVelocimeterSensor):
    def __init__(self, carla_instance : "RoarPyCarlaInstance", binded_target : RoarPyCarlaBase, name: str = "carla_local_velocimeter_sensor"):
        super().__init__(carla_instance, binded_target, name)
    
    async def receive_observation(self) -> RoarPyVelocimeterSensorData:
        velocity_global = self.binded_target.get_linear_3d_velocity()
        framequat = tr3d.euler.euler2quat(*self.binded_target.get_roll_pitch_yaw())
        framequat_inv = tr3d.quaternions.qinverse(framequat)
        velocity_local = tr3d.quaternions.rotate_vector(velocity_global, framequat_inv)

        self.received_data = RoarPyVelocimeterSensorData(
            velocity_local
        )
        return self.received_data
