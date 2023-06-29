import roar_py_interface
from roar_py_remote.services.base_stream_services import RoarPyStreamingClient
from roar_py_remote.worlds import RoarPyRemoteClientWorld, RoarPyRemoteServerWorldWrapper
from .base_stream_services import websocket_service
from ..worlds import RoarPyRemoteServerWorldManager, RoarPyRemoteMaskedWorld
from ..base import RoarPyObjectWithRemoteMessage
from .common import RoarPyRemoteClientWorldWithStreamClient
from typing import Union, TypeVar, Generic, Optional, Dict, Any
import websockets


class RoarPyWebsocketServerService(websocket_service.RoarPyWebsocketStreamingService):
    def __init__(self, server_world_manager : RoarPyRemoteServerWorldManager):
        super().__init__()
        self.server_world_manager = server_world_manager
        self._client_to_world_map : Dict[Any, RoarPyRemoteServerWorldWrapper] = {}
    
    async def initialize_streamed_world(self, client, world : roar_py_interface.RoarPyWorld) -> bool:
        return True

    async def generate_streamable_object(self, client) -> Optional[RoarPyObjectWithRemoteMessage]:
        masked_world = self.server_world_manager.get_world()
        return_world = await self.initialize_streamed_world(client, masked_world)
        masked_world = RoarPyRemoteServerWorldWrapper(masked_world)
        if return_world:
            self._client_to_world_map[client] = masked_world
            return masked_world
        else:
            return None

    async def client_disconnected(self, client):
        await super().client_disconnected(client)
        self._client_to_world_map[client].close()
        del self._client_to_world_map[client]
    
    async def __websocket_handler(self, websocket):
        await self.new_client_connected(websocket)
        while True:
            try:
                msg = await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                await self.client_disconnected(websocket)
                break
            except Exception as e:
                # print("Error while receiving message from client: {}".format(e))
                await self.client_disconnected(websocket)
                break

            await self.client_message_received(websocket, msg)

    async def run_server(self, bind_addr : str = "", bind_port : int = 80):
        async with websockets.serve(self.__websocket_handler, bind_addr, bind_port, max_size=None):
            while True:
                await self.server_world_manager._step()
                await self.tick()

class RoarPyWebsocketClientWorldWithStreamClient(RoarPyRemoteClientWorldWithStreamClient):
    def __init__(self, stream_service: RoarPyStreamingClient, websocket_client):
        super().__init__(stream_service)
        self._websocket_client = websocket_client
    
    async def try_receive_and_feed_service(self) -> None:
        try:
            msg = await self._websocket_client.recv()
        except websockets.exceptions.ConnectionClosed:
            self._stream_service.disconnected_from_server()
            raise ConnectionError("Connection closed while receiving message from server")
        except Exception as e:
            print("Error while receiving message from server: {}".format(e))
            return

        await self._stream_service.server_message_received(self._websocket_client, msg)


class RoarPyWebsocketClientService():
    @staticmethod
    async def connect_to_server(wss_address : str = "ws://localhost:8080") -> RoarPyWebsocketClientWorldWithStreamClient:
        websocket_client = await websockets.connect(wss_address, max_size=None)
        base_service = websocket_service.RoarPyWebsocketStreamingClient(RoarPyRemoteClientWorld)
        await base_service.connected_to_server(websocket_client)
        while base_service.stream_object is None:
            try:
                msg = await websocket_client.recv()
            except websockets.exceptions.ConnectionClosed:
                raise ConnectionError("Connection closed while initializing client")

            await base_service.server_message_received(websocket_client, msg)
        return RoarPyWebsocketClientWorldWithStreamClient(base_service, websocket_client)
