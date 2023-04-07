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
            self.close()
        finally:
            pass

    def get_last_gym_observation(self) -> typing.Optional[typing.Any]:
        return self.convert_obs_to_gym_obs(self.get_last_observation())
    
