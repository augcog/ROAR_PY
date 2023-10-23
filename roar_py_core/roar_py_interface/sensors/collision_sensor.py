from ..actors import RoarPyActor
from ..base import RoarPySensor, RoarPyRemoteSupportedSensorData
from ..base.sensor import remote_support_sensor_data_register
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
import typing

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyCollisionSensorData(RoarPyRemoteSupportedSensorData):
    # The actor the sensor is attached to, the one that measured the collision.
    actor: typing.Optional[RoarPyActor]
    # The second actor involved in the collision.
    other_actor: typing.Optional[typing.List[RoarPyActor]]
    # impulse (x,y,z local axis) in N*s
    impulse_normals: typing.Optional[typing.List[np.ndarray]] #np.NDArray[np.float32]

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low = -np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self):
        max_impulse_norm = -np.inf
        max_impulse = None
        for impulse_normal in self.impulse_normals:
            impulse_norm = np.linalg.norm(impulse_normal)
            if impulse_norm > max_impulse_norm:
                max_impulse_norm = impulse_norm
                max_impulse = impulse_normal
        return max_impulse


class RoarPyCollisionSensor(RoarPySensor[RoarPyCollisionSensorData]):
    sensordata_type = RoarPyCollisionSensorData
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
        return obs.convert_obs_to_gym_obs()
