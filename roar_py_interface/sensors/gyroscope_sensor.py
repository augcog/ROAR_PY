from .sensor import RoarPySensor
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

@serde
@dataclass
class RoarPyGyroscopeSensorData:
    # angular velocity (x,y,z local axis) in rad/s
    angular_velocity: np.NDArray[np.float32]

class RoarPyGyroscopeSensor(RoarPySensor[RoarPyGyroscopeSensorData]):
    def __init__(
        self, 
        name: str,
        control_timestep: float,
    ):
        super().__init__(name, control_timestep)

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )
    
    def convert_obs_to_gym_obs(self, obs: RoarPyGyroscopeSensorData):
        return obs.angular_velocity
