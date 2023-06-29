from roar_py_interface.worlds import RoarPyWorld
from roar_py_interface.wrappers.wrapper_base import RoarPyWrapper
from ..worlds import RoarPyRemoteClientWorld
from .base_stream_services import RoarPyStreamingClient
from roar_py_interface import RoarPyWorldWrapper, RoarPyWorld
from typing import TypeVar, Generic, Union, Any, Iterable

class RoarPyRemoteClientWorldWithStreamClient(RoarPyWorldWrapper):
    def __init__(self, stream_service : RoarPyStreamingClient):
        assert isinstance(stream_service.stream_object,RoarPyRemoteClientWorld)
        RoarPyWorldWrapper.__init__(self, stream_service.stream_object, "RoarPyRemoteClientWorldWithStreamClient")
        self._stream_service = stream_service
    
    async def try_receive_and_feed_service(self) -> None:
        raise NotImplementedError("This method is not implemented for RoarPyRemoteClientWorldWithStreamClient")
    
    async def step(self) -> float:
        await self._stream_service.tick() # Send out action commands for the next step
        await self.try_receive_and_feed_service()
        dt = await self._wrapped_object.step()
        return dt
