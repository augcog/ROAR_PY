import websockets
from roar_py_interface.base import RoarPyRemoteSupportedSensorData, RoarPySensor, RoarPyRemoteSupportedSensorSerializationScheme
from ..base import RoarPyObjectWithRemoteMessage
from serde import serde
from serde.msgpack import from_msgpack, to_msgpack
from dataclasses import dataclass
import enum
from typing import Callable, Optional, TypeVar, Generic, Dict, Type
import asyncio

_CommT = TypeVar("_CommT")
class RoarPyStreamingService(Generic[_CommT]):
    def __init__(self):
        self.client_to_stream_object : Dict[_CommT, RoarPyObjectWithRemoteMessage] = {}
    
    async def send_message_to_client(self, client: _CommT, message: bytes):
        pass

    async def disconnect_client(self, client: _CommT):
        pass

    async def generate_streamable_object(self, client: _CommT) -> RoarPyObjectWithRemoteMessage:
        pass
    
    async def new_client_connected(self, client: _CommT):
        new_streamable_object = await self.generate_streamable_object(client)
        self.client_to_stream_object[client] = new_streamable_object

        # Send the initial message
        packed_msg = new_streamable_object._pack_info()
        serialized_msg = to_msgpack(packed_msg)
        await self.send_message_to_client(client, serialized_msg)

    async def client_disconnected(self, client: _CommT):
        if client in self.client_to_stream_object:
            stream_object = self.client_to_stream_object[client]
            if hasattr(stream_object, "close"):
                stream_object.close()
            del self.client_to_stream_object[client]

    async def client_message_received(self, client: _CommT, message: bytes):
        if client not in self.client_to_stream_object:
            return
        stream_object = self.client_to_stream_object[client]
        try:
            msg_received = from_msgpack(stream_object._in_msg_type, message, strict_map_key=False)
        except Exception as e:
            print("Error Decoding Message",e)
            await self.disconnect_client(client)
            await self.client_disconnected(client)
            return

        stream_object._depack_info(msg_received)

    async def tick(self):
        tick_coroutines = []
        for client, stream_object in self.client_to_stream_object.items():
            if hasattr(stream_object, "receive_observation"):
                tick_coroutines.append(stream_object.receive_observation())
            elif hasattr(stream_object, "step"):
                tick_coroutines.append(stream_object.step())
        
        await asyncio.gather(*tick_coroutines)

        send_coroutines = []
        for client, stream_object in self.client_to_stream_object.items():
            packed_msg = stream_object._pack_info()
            serialized_msg = to_msgpack(packed_msg)
            send_coroutines.append(self.send_message_to_client(client, serialized_msg))
        
        await asyncio.gather(*send_coroutines, return_exceptions=True)


_CommClientT = TypeVar("_CommClientT")
class RoarPyStreamingClient(Generic[_CommClientT]):
    def __init__(self, target_type : Type[RoarPyObjectWithRemoteMessage]) -> None:
        super().__init__()
        self.target_type = target_type
        self.stream_object : Optional[RoarPyObjectWithRemoteMessage] = None
        self._connection : Optional[_CommClientT] = None

    async def send_message_to_server(self, connection : _CommClientT, message: bytes):
        pass

    async def disconnect_from_server(self, connection : _CommClientT):
        pass

    async def disconnected_from_server(self):
        self._connection = None

    async def connected_to_server(self, connection : _CommClientT):
        self._connection = connection

    async def server_message_received(self, connection : _CommClientT, message: bytes):
        try:
            msg_received = from_msgpack(self.target_type._in_msg_type, message, strict_map_key=False)
        except Exception as e:
            print("Error Decoding Message",e)
            await self.disconnect_from_server(connection)
            await self.disconnected_from_server()
            return
        
        if self.stream_object is None:
            self._connection = connection
            self.stream_object = self.target_type(msg_received)
        else:
            self.stream_object._depack_info(msg_received)

    async def tick(self):
        if self.stream_object is None or self._connection is None:
            return
        packed_msg = self.stream_object._pack_info()
        serialized_msg = to_msgpack(packed_msg)
        await self.send_message_to_server(self._connection,serialized_msg)
