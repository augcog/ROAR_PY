import roar_py_carla_implementation
import roar_py_interface
import carla
import pygame
from PIL import Image
import numpy as np
import asyncio

class ManualControlViewer:
    def __init__(
        self
    ):
        self.screen = None
    
    def init_pygame(self, x, y) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Manual Control Viewer")

    def render(self, image : roar_py_interface.RoarPyCameraSensorData) -> None:
        image_pil : Image = image.get_image()
        if self.screen is None:
            self.init_pygame(image_pil.width, image.height)
        
        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()


async def main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla_implementation.RoarPyCarlaInstance(carla_client)
    
    carla_world = roar_py_instance.world
    carla_world.set_asynchronous(True)
    carla_world.set_control_steps(0.05, 0.005)

    vehicle = carla_world.spawn_vehicle(
        "vehicle.tesla.model3",
        np.zeros(3),
        np.zeros(3),
        True        
    )

    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.zeros(3),
        np.zeros(3)
    )

    while True:
        img = await camera.receive_observation()
        ManualControlViewer().render(img)

if __name__ == '__main__':
    asyncio.run(main())