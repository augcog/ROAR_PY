from roar_py_interface import RoarPySensor, RoarPyRemoteSupportedSensorData
from typing import Any, TypeVar, Generic

_ObsT = TypeVar("_ObsT")
class RoarPyRemoteSharedSensor(RoarPySensor[_ObsT], Generic[_ObsT]):
    async def receive_observation(self) -> Any:
        raise PermissionError("This sensor is shared, you cannot receive observation from it")
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        raise PermissionError("Cannot set attribute on a shared actor")

    def __delattr__(self, __name: str) -> None:
        raise PermissionError("Cannot delete attribute on a shared actor")

    def close(self):
        pass
    