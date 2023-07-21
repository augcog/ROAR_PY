import typing
from ..actors.actor import RoarPyActor
from ..base import RoarPySensor
from .waypoint import RoarPyWaypoint

class RoarPyWorld:
    @property
    def is_asynchronous(self):
        raise NotImplementedError
    
    def get_actors(self) -> typing.Iterable[RoarPyActor]:
        raise NotImplementedError

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        raise NotImplementedError
    
    @property
    def maneuverable_waypoints(self) -> typing.Optional[typing.Iterable[RoarPyWaypoint]]:
        return None
    
    @property
    def comprehensive_waypoints(self) -> typing.Optional[typing.Dict[typing.Any, typing.List[RoarPyWaypoint]]]:
        return None

    @property
    def last_tick_elapsed_seconds(self) -> float:
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
