from roar_py_interface.actors.vehicle import *
from roar_py_interface.sensors import *
from roar_py_interface.wrappers import roar_py_thread_sync
import typing
import gymnasium as gym
import carla
import transforms3d as tr3d
from .carla_actor import RoarPyCarlaActor

class RoarPyCarlaVehicle(RoarPyCarlaActor):
    def __init__(self, carla_instance : "RoarPyCarlaInstance", carla_actor: carla.Vehicle, auto_gear : bool = False, name : str = "carla_vehicle"):
        RoarPyCarlaActor.__init__(self, carla_instance, carla_actor, name=name)
        self.auto_gear = auto_gear
    
    @property
    @roar_py_thread_sync
    def num_gears(self) -> int:
        return len(self.carla_actor.get_physics_control().forward_gears)

    def get_action_spec(self) -> gym.Space:
        spec_dict = {
            "throttle": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            "steer": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
            "brake": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            "hand_brake": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            "reverse": gym.spaces.MultiBinary(1)
        }
        if not self.auto_gear:
            spec_dict["target_gear"] = gym.spaces.Discrete(self.num_gears,start=1)
        return gym.spaces.Dict(spec_dict)
    
    @staticmethod
    def translate_action_to_carla_vehicle_control(is_autogear : bool, action : typing.Dict[str,typing.Any]) -> carla.VehicleControl:
        control = carla.VehicleControl()
        control.throttle = float(action["throttle"])
        control.steer = float(action["steer"])
        control.brake = float(action["brake"])
        control.hand_brake = float(action["hand_brake"]) >= 0.5
        control.reverse = bool(action["reverse"])
        if not is_autogear:
            control.manual_gear_shift = True
            control.gear = action["target_gear"]
        else:
            control.manual_gear_shift = False
            control.gear = 0
        return control

    @roar_py_thread_sync
    async def _apply_action(self, action: typing.Any) -> bool:
        control = self.translate_action_to_carla_vehicle_control(self.auto_gear,action)
        self._base_actor.apply_control(control)
        return True