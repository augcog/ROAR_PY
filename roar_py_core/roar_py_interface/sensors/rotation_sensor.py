from ..base import RoarPySensor, RoarPyRemoteSupportedSensorData
from ..base.sensor import remote_support_sensor_data_register
from serde import serde
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
import transforms3d as tr3d
import typing

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyFrameQuatSensorData(RoarPyRemoteSupportedSensorData):
    # Normalized frame quaternion (w,x,y,z)
    frame_quaternion: np.ndarray #np.NDArray[np.float32]

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
    
    def convert_obs_to_gym_obs(self):
        return self.frame_quaternion

class RoarPyFrameQuatSensor(RoarPySensor[RoarPyFrameQuatSensorData]):
    sensordata_type = RoarPyFrameQuatSensorData
    def __init__(
        self, 
        control_timestep: float,
        name: str = "framequat_sensor",
    ):
        super().__init__(name, control_timestep)

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

    def convert_obs_to_gym_obs(self, obs: RoarPyFrameQuatSensorData):
        return obs.convert_obs_to_gym_obs()

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyRollPitchYawSensorData(RoarPyRemoteSupportedSensorData):
    """
    Normalized[-pi,pi) roll,pitch,yaw
    """
    roll_pitch_yaw: np.ndarray #np.NDArray[np.float32]

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =-np.pi,
            high=np.pi,
            shape=(3,),
            dtype=np.float32
        )
    
    def convert_obs_to_gym_obs(self):
        return self.roll_pitch_yaw
    
class RoarPyRollPitchYawSensor(RoarPySensor[RoarPyRollPitchYawSensorData]):
    sensordata_type = RoarPyRollPitchYawSensorData
    def __init__(
        self, 
        name: str,
        control_timestep: float,
    ):
        super().__init__(name, control_timestep)

    def get_gym_observation_spec(self) -> gym.Space:
        return gym.spaces.Box(
            low =-np.pi,
            high=np.pi,
            shape=(3,),
            dtype=np.float32
        )
    
    def convert_obs_to_gym_obs(self, obs: RoarPyRollPitchYawSensorData):
        return obs.convert_obs_to_gym_obs()

class RoarPyRollPitchYawSensorFromFrameQuat(RoarPyRollPitchYawSensor):
    def __init__(
        self,
        framequat_sensor: RoarPyFrameQuatSensor,
        name: typing.Optional[str] = None
    ):
        super().__init__(
            name=framequat_sensor.name+" -> rpy" if name is None else name,
            control_timestep=framequat_sensor.control_timestep
        )
        self.framequat_sensor = framequat_sensor
        self._last_obs = None
    
    async def receive_observation(self) -> RoarPyRollPitchYawSensorData:
        dat : RoarPyFrameQuatSensorData = await self.framequat_sensor.receive_observation()
        obs = np.array(tr3d.euler.quat2euler(dat.frame_quaternion))
        self._last_obs = RoarPyRollPitchYawSensorData(obs)
        return self._last_obs
    
    def get_last_observation(self) -> typing.Optional[RoarPyRollPitchYawSensorData]:
        return self._last_obs
    
    def close(self):
        return self.framequat_sensor.close()
    
    def is_closed(self) -> bool:
        return self.framequat_sensor.is_closed()
    
class RoarPyFrameQuatSensorFromRollPitchYaw(RoarPyFrameQuatSensor):
    def __init__(
        self,
        rpy_sensor: RoarPyRollPitchYawSensor,
        name: typing.Optional[str] = None
    ):
        super().__init__(
            name=rpy_sensor.name+" -> framequat" if name is None else name,
            control_timestep=rpy_sensor.control_timestep
        )
        self.rpy_sensor = rpy_sensor
        self._last_obs = None
    
    async def receive_observation(self) -> RoarPyFrameQuatSensorData:
        dat : RoarPyRollPitchYawSensorData = await self.rpy_sensor.receive_observation()
        obs = tr3d.euler.euler2quat(*dat.roll_pitch_yaw)
        self._last_obs = RoarPyFrameQuatSensorData(obs)
        return self._last_obs
    
    def get_last_observation(self) -> typing.Optional[RoarPyFrameQuatSensorData]:
        return self._last_obs
    
    def close(self):
        return self.rpy_sensor.close()
    
    def is_closed(self) -> bool:
        return self.rpy_sensor.is_closed()