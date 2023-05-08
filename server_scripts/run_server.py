import carla
import roar_py_carla_implementation
import roar_py_remote
import rpyc
import threading
import asyncio

IS_ASYNC = False

class RoarPyServer(rpyc.Service):
    def __init__(self) -> None:
        super().__init__()
        carla_client = carla.Client('localhost', 2000)
        carla_client.set_timeout(5.0)
        
        roar_py_instance = roar_py_carla_implementation.RoarPyCarlaInstance(carla_client)
        roar_py_instance.world.set_asynchronous(False)
        roar_py_instance.world.set_control_steps(0.05, 0.005)
        self._server_world = roar_py_remote.RoarPyRemoteServer(roar_py_instance.world, IS_ASYNC)

    def get_server_world(self) -> roar_py_remote.RoarPyRemoteServer:
        return self._server_world

if __name__ == '__main__':
    from rpyc.utils.server import ThreadPoolServer
    print("Initializing server...")
    service = RoarPyServer()
    print("Carla Server initialized.")
    server = ThreadPoolServer(service, port=18861, protocol_config={"allow_public_attrs": True})
    server_world : roar_py_remote.RoarPyRemoteServer = service.get_server_world()
    
    def step_world_runner():
        async def event_loop():
            while True:
                await server_world._step()
        asyncio.run(event_loop())
    
    step_world_process = threading.Thread(target=step_world_runner, daemon=True)
    step_world_process.start()

    server.start()
