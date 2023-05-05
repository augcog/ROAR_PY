import typing
import asyncio
import gymnasium as gym

_ObsT = typing.TypeVar("_ObsT")
class RoarPySensor(typing.Generic[_ObsT]):
    def __init__(
        self, 
        name: str,
        control_timestep: float,
    ):
        self.name = name
        self.control_timestep = control_timestep

    def get_gym_observation_spec(self) -> gym.Space:
        raise NotImplementedError()
    
    async def receive_observation(self) -> _ObsT:
        raise NotImplementedError()
    
    def get_last_observation(self) -> typing.Optional[_ObsT]:
        raise NotImplementedError()

    def convert_obs_to_gym_obs(self, obs: _ObsT):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()
    
    def is_closed(self) -> bool:
        raise NotImplementedError()

    def __del__(self):
        try:
            if not self.is_closed():
                self.close()
        finally:
            pass

    def get_last_gym_observation(self) -> typing.Optional[typing.Any]:
        return self.convert_obs_to_gym_obs(self.get_last_observation())

def custom_roar_py_sensor_wrapper(
    sensor: RoarPySensor,
    gym_observation_spec_override: typing.Optional[gym.Space],
    close_lambda: typing.Optional[typing.Callable[[RoarPySensor], None]],
    receive_observation_lambda: typing.Optional[typing.Callable[[RoarPySensor], typing.Any]],
    convert_obs_to_gym_obs_lambda: typing.Optional[typing.Callable[[RoarPySensor, typing.Any], typing.Any]],
):
    if gym_observation_spec_override is not None:
        sensor.get_gym_observation_spec = lambda: gym_observation_spec_override
    
    if close_lambda is not None:
        sensor.closed = False
        def custom_close():
            close_lambda(sensor)
            sensor.closed = True
        sensor.close = custom_close
        sensor.is_closed = lambda: sensor.closed

    if receive_observation_lambda is not None:
        sensor.last_obs = None
        def custom_receive_obs():
            sensor.last_obs = receive_observation_lambda(sensor)
            return sensor.last_obs
        sensor.receive_observation = custom_receive_obs
        sensor.get_last_observation = lambda: sensor.last_obs

    if convert_obs_to_gym_obs_lambda is not None:
        sensor.convert_obs_to_gym_obs = lambda obs: convert_obs_to_gym_obs_lambda(sensor, obs)