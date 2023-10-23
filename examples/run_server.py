import carla
import roar_py_remote
import roar_py_interface
import roar_py_remote.services.base_stream_services.websocket_service
import roar_py_carla
import threading
import asyncio
from typing import Union
import numpy as np
import websockets

IS_ASYNC = False
SYNC_WAIT_TIME = 0.5


class ServerInst(roar_py_remote.RoarPyWebsocketServerService):
    async def initialize_streamed_world(self, client, world : roar_py_carla.RoarPyCarlaWorld) -> bool:
        # This method is called when a client connects to the server and requests a streamed world.
        # You can use this method to initialize the world for the client.
        
        waypoints = world.maneuverable_waypoints
        new_vehicle = None
        while new_vehicle is None:
            spawning_waypoint = waypoints[0]
            new_vehicle = world.spawn_vehicle(
                "vehicle.tesla.model3",
                spawning_waypoint.location + np.array([0,0,0.5]), # spawn 1.5m above the ground
                spawning_waypoint.roll_pitch_yaw,
                True,
                "roar_py_remote_vehicle"
            )
        new_vehicle.attach_camera_sensor(
            roar_py_interface.RoarPyCameraSensorDataRGB, # Specify what kind of data you want to receive
            np.array([-2.0 * new_vehicle.bounding_box.extent[0], 0.0, 3.0 * new_vehicle.bounding_box.extent[2]]), # relative position
            np.array([0, 10/180.0*np.pi, 0]), # relative rotation
            image_width=600,
            image_height=400
        )
        await world.step() # Make sure the world is updated before we start using it
        return True

async def _main():
    print("Connecting to Carla...")
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(5.0)
    
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    roar_py_instance.world.set_asynchronous(True)
    roar_py_instance.world.set_control_steps(0.0, 0.005)
    await roar_py_instance.world.step()
    roar_py_instance.clean_actors_not_registered()

    roar_py_server_manager = roar_py_remote.RoarPyRemoteServerWorldManager(roar_py_instance.world, IS_ASYNC, SYNC_WAIT_TIME, thread_safe=False)

    print("Initializing websocket server...")
    roar_py_websocket_server = ServerInst(roar_py_server_manager)
    print("Initialized, starting websocket server...")
    try:
        await roar_py_websocket_server.run_server(bind_port=8080)
    finally:
        roar_py_instance.close()

if __name__ == "__main__":
    asyncio.run(_main())