from .actor import RoarPyActor
import gymnasium as gym

class RoarPyVehicleAutoGearActor(RoarPyActor):
    def __init__(
        self,
        name: str = "vehicle_autogear",
        control_timestep : float = 0.05,
        force_real_control_timestep : bool = False
    ):
        super().__init__(
            name=name,
            control_timestep=control_timestep,
            force_real_control_timestep=force_real_control_timestep
        )

    def get_action_spec(self) -> gym.Space:
        return gym.spaces.Dict({
            "throttle": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            "steer": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
            "brake": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            "hand_brake": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            "reverse": gym.spaces.MultiBinary(1)
        })

class RoarPyVehicleManualGearActor(RoarPyActor):
    def __init__(
        self,
        name: str = "vehicle_manualgear",
        control_timestep : float = 0.05,
        force_real_control_timestep : bool = False,
        num_gears: int = 6
    ):
        super().__init__(
            name=name,
            control_timestep=control_timestep,
            force_real_control_timestep=force_real_control_timestep
        )
        self.num_gears = num_gears

    def get_action_spec(self) -> gym.Space:
        return gym.spaces.Dict({
            "throttle": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            "steer": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
            "brake": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            "hand_brake": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
            "reverse": gym.spaces.MultiBinary(1),
            "target_gear": gym.spaces.Discrete(self.num_gears,start=1)
        })