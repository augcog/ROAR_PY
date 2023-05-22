import rpyc
from roar_py_interface import RoarPyWorld
from ..worlds import RoarPyRemoteServer

class RoarPyRPYCService(rpyc.Service):
    def __init__(self, remote_server:RoarPyRemoteServer) -> None:
        super().__init__()
        self._server_world = remote_server

    @rpyc.exposed
    def get_remote_server(self) -> RoarPyRemoteServer:
        return self._server_world
