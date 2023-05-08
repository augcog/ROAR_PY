from typing import Any
from roar_py_interface.actors.actor import RoarPyActor, RoarPyResettableActor
from roar_py_interface.sensors import *
import typing
import gymnasium as gym
import carla
import transforms3d as tr3d
import numpy as np

class RoarPyRemoteSharedActor(RoarPyActor):
    async def apply_action(self, action: typing.Any) -> bool:
        return False # No action should be applied if this is a shared actor
    
    async def receive_observation(self) -> typing.Dict[str,typing.Any]:
        raise PermissionError("Cannot receive observation from a shared actor")

    def __setattr__(self, __name: str, __value: Any) -> None:
        raise PermissionError("Cannot set attribute on a shared actor")

    def __delattr__(self, __name: str) -> None:
        raise PermissionError("Cannot delete attribute on a shared actor")

    def close(self) -> None:
        pass

    async def reset(self) -> None:
        if isinstance(self._wrapped_object, RoarPyResettableActor):
            raise PermissionError("Cannot reset a shared actor")
        else:
            raise PermissionError("Cannot reset a non-resettable actor")
