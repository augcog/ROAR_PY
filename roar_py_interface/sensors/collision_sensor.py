from .sensor import RoarPySensor
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
import typing

@serde
@dataclass
class RoarPyCollisionSensorData:
    # The actor the sensor is attached to, the one that measured the collision.
    actor: typing.Optional[typing.Any]
    # The second actor involved in the collision.
    other_actor: typing.Optional[typing.Any]
    # impulse (x,y,z local axis) in N*s
    impulse_normal: np.NDArray[np.float32]


class RoarPyCollisionSensor(RoarPySensor[RoarPyCollisionSensorData]):
    def __init__(
        self, 
        name: str,
        control_timestep: float,
    ):
        super().__init__(name, control_timestep)

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low = -np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self, obs: RoarPyCollisionSensorData):
        return obs.impulse_normal
    
    @property
    def other_actor(self) -> typing.Optional[typing.Any]:
        raise NotImplementedError()
    
    @property
    def impulse_normal(self) -> np.array:
        raise NotImplementedError()
