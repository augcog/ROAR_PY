from roar_py_interface.actors.actor import RoarPyActor, RoarPyResettableActor
from roar_py_interface.sensors import *
import typing
import gymnasium as gym
import carla
import transforms3d as tr3d

class RoarPyCarlaActor(RoarPyActor):
    def __init__(
        self, 
        carla_actor: carla.Actor,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.carla_actor = carla_actor
        self._internal_sensors = []
        self.frame_quat_sensor = None

    def get_action_spec(self) -> gym.Space:
        raise NotImplementedError()
    
    async def __apply_action(self, action: typing.Any) -> bool:
        raise NotImplementedError()
    
    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        return self._internal_sensors
    
    def is_framequat_sensor_enabled(self) -> bool:
        exist = False
        for sensor in self._internal_sensors:
            if isinstance(sensor, RoarPyFrameQuatSensor):
                exist = True
                break
        return exist

    @staticmethod
    def __framequat_sensor_obs_fn(sensor: RoarPyFrameQuatSensor) -> RoarPyFrameQuatSensorData:
        rot = sensor.actor_native.get_transform().rotation
        quaternion = tr3d.euler.euler2quat(rot.roll, rot.pitch, rot.yaw)
        return RoarPyFrameQuatSensorData(quaternion)

    def set_framequat_sensor_enabled(self, enabled: bool = True):
        if enabled and not self.is_framequat_sensor_enabled():
            ss = RoarPyFrameQuatSensor(0.05)
            ss.actor_native = self.carla_actor
            framequat_sensor = custom_roar_py_sensor_wrapper(
                ss,
                gym_observation_spec_override=None,
                close_lambda=lambda sensor: None,
                receive_observation_lambda=__class__.__framequat_sensor_obs_fn,
                convert_obs_to_gym_obs_lambda=None
            )
            self._internal_sensors.append(framequat_sensor)
        elif not enabled and self.is_framequat_sensor_enabled():
            for sensor in self._internal_sensors:
                if isinstance(sensor, RoarPyFrameQuatSensor):
                    sensor.close()
                    self._internal_sensors.remove(sensor)
                    break

    def close(self):
        if self.carla_actor.is_alive:
            self.carla_actor.destroy()

    def is_closed(self) -> bool:
        return self.carla_actor.is_alive