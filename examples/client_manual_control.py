import roar_py_interface
import roar_py_remote
import asyncio
from manual_control import ManualControlViewer

WEBSOCKET_URI = "ws://localhost:8080"

async def main():
    world = await roar_py_remote.RoarPyWebsocketClientService.connect_to_server(WEBSOCKET_URI)
        
    vehicle : roar_py_interface.RoarPyActor = world.get_actors()[0]
    camera : roar_py_interface.RoarPySensor = vehicle.get_sensors()[0]

    viewer = ManualControlViewer()

    while True:
        await world.step()
        img : roar_py_interface.RoarPyCameraSensorDataRGB = await camera.receive_observation()
        control = viewer.render(img)
        if control is None:
            break
        await vehicle.apply_action(control)
            
if __name__ == '__main__':
    asyncio.run(main())