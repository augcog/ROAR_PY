import roar_py_carla_implementation
import roar_py_interface
import roar_py_remote
import roar_py_remote.services.websocket_service
import carla
import pygame
from PIL import Image
import numpy as np
import asyncio
from typing import Optional, Dict, Any
import websockets

class ManualControlViewer:
    def __init__(
        self
    ):
        self.screen = None
        self.clock = None
        self.last_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0])
        }
    
    def init_pygame(self, x, y) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Manual Control Viewer")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()

    def render(self, image : roar_py_interface.RoarPyCameraSensorData) -> Optional[Dict[str, Any]]:
        image_pil : Image = image.get_image()
        if self.screen is None:
            self.init_pygame(image_pil.width, image_pil.height)
        
        new_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0])
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_UP]:
            new_control['throttle'] = 0.5
        if pressed_keys[pygame.K_DOWN]:
            new_control['brake'] = 0.2
        if pressed_keys[pygame.K_LEFT]:
            new_control['steer'] = -0.2
        if pressed_keys[pygame.K_RIGHT]:
            new_control['steer'] = 0.2
        
        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        self.screen.fill((0,0,0))
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)
        self.last_control = new_control
        return new_control

WEBSOCKET_URI = "ws://localhost:8080"

async def main():
    async with websockets.connect(WEBSOCKET_URI, max_size=None) as websocket_client:
        roar_py_remote_client = roar_py_remote.services.websocket_service.RoarPyWebsocketStreamingClient(
            roar_py_remote.RoarPyRemoteClientActor
        ) # Receive a client that can be used to send and receive messages from the server
        while roar_py_remote_client.stream_object is None:
            try:
                msg = await websocket_client.recv()
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed while initializing client")
                return
            await roar_py_remote_client.server_message_received(websocket_client, msg)
        
        vehicle : roar_py_interface.RoarPyActor = roar_py_remote_client.stream_object
        camera : roar_py_interface.RoarPySensor = vehicle.get_sensors()[0]

        viewer = ManualControlViewer()

        while True:
            # img.get_image().save("test.png")
            try:
                msg = await websocket_client.recv()
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed while receiving")
                await roar_py_remote_client.disconnected_from_server()
                return
            await roar_py_remote_client.server_message_received(websocket_client, msg)
            

            img : roar_py_interface.RoarPyCameraSensorDataRGB = await camera.receive_observation()
            control = viewer.render(img)
            if control is None:
                break
            await vehicle.apply_action(control)

            await roar_py_remote_client.tick()
            
if __name__ == '__main__':
    asyncio.run(main())