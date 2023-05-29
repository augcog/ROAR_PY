import roar_py_carla_implementation
import roar_py_interface
import carla
import pygame
from PIL import Image
import numpy as np
import asyncio
from typing import Optional, Dict, Any

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
            new_control['throttle'] = 1.0
        if pressed_keys[pygame.K_DOWN]:
            new_control['brake'] = 1.0
        if pressed_keys[pygame.K_LEFT]:
            new_control['steer'] = -1.0
        if pressed_keys[pygame.K_RIGHT]:
            new_control['steer'] = 1.0
        
        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        self.screen.fill((0,0,0))
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)
        self.last_control = new_control
        return new_control


async def main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla_implementation.RoarPyCarlaInstance(carla_client)
    
    carla_world = roar_py_instance.world
    carla_world.set_control_steps(0.05, 0.005)
    carla_world.set_asynchronous(True)
    
    spawn_point, spawn_rpy = carla_world.spawn_points[
        np.random.randint(len(carla_world.spawn_points))
    ]

    vehicle = carla_world.spawn_vehicle(
        "vehicle.tesla.model3",
        spawn_point + np.array([0, 0, 0.5]),
        spawn_rpy
    )

    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB, # Specify what kind of data you want to receive
        np.array([0.5, 0.0, 9.5]), # relative position
        np.array([0, -np.pi/2, 0]), # relative rotation
    )

    viewer = ManualControlViewer()

    try:
        while True:
            # img.get_image().save("test.png")
            await carla_world.step()
            img : roar_py_interface.RoarPyCameraSensorDataRGB = await camera.receive_observation()
            control = viewer.render(img)
            if control is None:
                break
            await vehicle.apply_action(control)
    finally:
        roar_py_instance.close()

if __name__ == '__main__':
    asyncio.run(main())