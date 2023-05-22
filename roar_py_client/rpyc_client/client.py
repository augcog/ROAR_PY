from roar_py_remote import RoarPyRemoteServer, RoarPyRPYCService, RoarPyRemoteMaskedWorld
from roar_py_interface import RoarPyWorld
from typing import TypeVar, Generic, Optional, List, Type, Callable, Any, Union
import rpyc

_WorldType = TypeVar("_WorldType", bound=RoarPyWorld)
class RoarPyRPYCClient(Generic[_WorldType]):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 18861,
    ):
        self.rpyc_client = rpyc.connect(
            host,port,config={
                "sync_request_timeout": None
            },keepalive=True
        )
        self._remote_world = None
    
    def get_rpyc_service(self) -> RoarPyRPYCService:
        return self.rpyc_client.root
    
    def get_world(self) -> Union[_WorldType, RoarPyRemoteMaskedWorld]:
        if self._remote_world is None:
            self._remote_world = self.get_rpyc_service().get_remote_server().get_world()
        return self._remote_world