from roar_py_interface.actors.actor import RoarPyActor, RoarPyResettableActor
from roar_py_interface.sensors import *
import typing
import gymnasium as gym
import carla
import transforms3d as tr3d
from ..base import RoarPyCarlaBase
from ..clients import RoarPyCarlaInstance

class RoarPyCarlaActor(RoarPyActor, RoarPyCarlaBase):
    def __init__(
        self, 
        carla_instance: RoarPyCarlaInstance,
        carla_actor: carla.Actor,
        *args,
        **kwargs
    ):
        RoarPyActor.__init__(self, *args, **kwargs)
        RoarPyCarlaBase.__init__(self, carla_instance, carla_actor)
        self._internal_sensors = []
        self.frame_quat_sensor = None

    def get_action_spec(self) -> gym.Space:
        raise NotImplementedError()
    
    async def __apply_action(self, action: typing.Any) -> bool:
        raise NotImplementedError()
    
    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        return self._internal_sensors