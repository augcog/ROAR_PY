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
class RoarPyRadarSensorData(RoarPyRemoteSupportedSensorData):
    # Received list of 4D points in shape (N, (altitude, azimuth, depth, velocity))
    # where N is the # of points,
    # altitude is the altitude angle of the detection (float, radians)
    # azimuth is the azimuth angle of the detection (float, radians)
    # depth is the depth of the detection (float, meters)
    # velocity is the velocity of the detection (float, meters per second)
    radar_points_data: np.ndarray

    def get_gym_observation_spec(self) -> gym.Space:
        N = self.radar_points_data.shape[0]
        return gym.spaces.Box(
            low = np.array([[-np.inf, -np.inf, -np.inf, -np.inf]]),
            high= np.array([[ np.inf,  np.inf,  np.inf,  np.inf]]),
            shape=(N, 4),
            dtype=np.float32
        )
    
    def convert_obs_to_gym_obs(self):
        return self.radar_points_data
    
class RoarPyRadarSensor(RoarPySensor[RoarPyRadarSensorData]):
    sensordata_type = RoarPyRadarSensorData
    def __init__(
        self,
        name: str,
        control_timestep: float
    ):
        super().__init__(name, control_timestep)

    @property
    def horizontal_fov(self) -> float:
        raise NotImplementedError
    
    @property
    def points_per_second(self) -> float:
        raise NotImplementedError
    
    @property
    def max_distance(self) -> float:
        raise NotImplementedError
    
    @property
    def vertical_fov(self) -> float:
        raise NotImplementedError

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10000, 4), # TODO: Fix this later
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self, obs: RoarPyRadarSensorData):
        return obs.convert_obs_to_gym_obs()
