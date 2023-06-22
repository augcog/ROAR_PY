from .base_service import RoarPyStreamingService, RoarPyStreamingClient
import websockets

class RoarPyWebsocketStreamingService(RoarPyStreamingService):
    async def send_message_to_client(self, client, message: bytes):
        await client.send(message)

    async def disconnect_client(self, client):
        await client.close()

class RoarPyWebsocketStreamingClient(RoarPyStreamingClient):
    async def send_message_to_server(self, connection, message: bytes):
        await connection.send(message)

    async def disconnect_from_server(self, connection):
        await connection.close()