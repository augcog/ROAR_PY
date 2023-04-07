import typing
from ..actors.actor import RoarPyActor
from ..sensors.sensor import RoarPySensor

class RoarPyWorld:
    @property
    def is_asynchronous(self):
        raise NotImplementedError
    
    def get_actors(self) -> typing.Iterable[RoarPyActor]:
        raise NotImplementedError

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        raise NotImplementedError

    """
    Steps the world by one step.
    This can be used to update the world state and inside actor/sensor states
    Returns the dt of the step compared to previous step.
    """
    async def step(self) -> float:
        raise NotImplementedError

class RoarPyWorldResettable(RoarPyWorld):
    """
    Resets the world(and its containing actors/sensor) to its initial state
    """
    async def reset(self):
        raise NotImplementedError
