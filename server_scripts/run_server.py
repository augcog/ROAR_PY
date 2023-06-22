import carla
import roar_py_carla_implementation
import roar_py_remote
import roar_py_interface
import roar_py_remote.services.websocket_service
import threading
import asyncio
from typing import Union
import numpy as np
import websockets

IS_ASYNC = False
SYNC_WAIT_TIME = 0.5


class RoarPyWebsocketServerImpl(roar_py_remote.services.websocket_service.RoarPyWebsocketStreamingService):
    def __init__(self, server_world_manager : roar_py_remote.RoarPyRemoteServerWorldManager):
        super().__init__()
        self.__server_world_manager = server_world_manager
        self._client_to_world_map = {}
    
    async def generate_streamable_object(self, client) -> roar_py_remote.RoarPyObjectWithRemoteMessage:
        masked_world : roar_py_carla_implementation.RoarPyCarlaWorld = self.__server_world_manager.get_world()
        waypoints = masked_world.maneuverable_waypoints

        self._client_to_world_map[client] = masked_world
        
        new_vehicle = None
        while new_vehicle is None:
            spawning_waypoint = waypoints[np.random.randint(len(waypoints))]
            new_vehicle = masked_world.spawn_vehicle(
                "vehicle.tesla.model3",
                spawning_waypoint.location + np.array([0,0,1.5]), # spawn 1.5m above the ground
                spawning_waypoint.roll_pitch_yaw,
                True,
                "roar_py_remote_vehicle"
            )
        
        camera = new_vehicle.attach_camera_sensor(
            roar_py_interface.RoarPyCameraSensorDataRGB, # Specify what kind of data you want to receive
            np.array([-0.5, 0.0, 3.5]), # relative position
            np.array([0, np.pi/10, 0]), # relative rotation
            image_width=1024,
            image_height=768
        )
        
        return roar_py_remote.RoarPyRemoteServerActorWrapper(new_vehicle)

    async def client_disconnected(self, client):
        await super().client_disconnected(client)
        self._client_to_world_map[client].close()
        del self._client_to_world_map[client]

async def _main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(5.0)
    
    roar_py_instance = roar_py_carla_implementation.RoarPyCarlaInstance(carla_client)
    roar_py_instance.world.set_asynchronous(True)
    roar_py_instance.world.set_control_steps(0.0, 0.005)

    roar_py_server_manager = roar_py_remote.RoarPyRemoteServerWorldManager(roar_py_instance.world, IS_ASYNC, SYNC_WAIT_TIME)

    print("Initializing websocket server...")
    service = RoarPyWebsocketServerImpl(roar_py_server_manager)
    async def websocket_handler(websocket):
        await service.new_client_connected(websocket)
        for msg in await websocket:
            await service.client_message_received(websocket, msg)
    
    # def step_world_runner():
    #     async def event_loop():
    #         while True:
    #             await asyncio.sleep(0.1)
    #             await roar_py_server._step()
    #     asyncio.run(event_loop())
    
    # print("Setting up step world process...")
    # step_world_process = threading.Thread(target=step_world_runner, daemon=True)
    # step_world_process.start()
    try:
        async with websockets.serve(websocket_handler, "", 8080):
            while True:
                await roar_py_server_manager._step()
                await service.tick()
    finally:
        roar_py_instance.close()

if __name__ == "__main__":
    asyncio.run(_main())