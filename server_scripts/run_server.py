import carla
import roar_py_carla_implementation
import roar_py_remote
import rpyc
import threading
import asyncio

IS_ASYNC = False
SYNC_WAIT_TIME = 0.5

if __name__ == '__main__':
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(5.0)
    
    roar_py_instance = roar_py_carla_implementation.RoarPyCarlaInstance(carla_client)
    roar_py_instance.world.set_asynchronous(True)
    roar_py_instance.world.set_control_steps(0.0, 0.005)

    roar_py_server = roar_py_remote.RoarPyRemoteServer(roar_py_instance.world, IS_ASYNC, SYNC_WAIT_TIME)

    from rpyc.utils.server import ThreadPoolServer
    print("Initializing server...")
    service = roar_py_remote.RoarPyRPYCService(roar_py_server)
    print("Carla World initialized.")
    server = ThreadPoolServer(service, port=18861, protocol_config={"allow_public_attrs": True})
    roar_py_server : roar_py_remote.RoarPyRemoteServer = service.get_remote_server()
    
    def step_world_runner():
        async def event_loop():
            while True:
                await asyncio.sleep(0.1)
                await roar_py_server._step()
        asyncio.run(event_loop())
    
    print("Setting up step world process...")
    step_world_process = threading.Thread(target=step_world_runner, daemon=True)
    step_world_process.start()

    print("Starting rpyc server...")
    server.start()
    roar_py_instance.close()