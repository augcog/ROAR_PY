import roar_py_carla
import roar_py_interface
import carla
import pygame
from PIL.Image import Image
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

    def render(self, image : roar_py_interface.RoarPyCameraSensorData, occupancy_map : Optional[Image] = None) -> Optional[Dict[str, Any]]:
        image_pil : Image = image.get_image()
        occupancy_map_rgb = occupancy_map.convert("RGB") if occupancy_map is not None else None
        if self.screen is None:
            if occupancy_map_rgb is None:
                self.init_pygame(image_pil.width, image_pil.height)
            else:
                self.init_pygame(image_pil.width + occupancy_map.width, image_pil.height)
        
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
            new_control['throttle'] = 0.4
        if pressed_keys[pygame.K_DOWN]:
            new_control['brake'] = 0.2
        if pressed_keys[pygame.K_LEFT]:
            new_control['steer'] = -0.2
        if pressed_keys[pygame.K_RIGHT]:
            new_control['steer'] = 0.2
        
        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        if occupancy_map_rgb is not None:
            occupancy_map_surface = pygame.image.fromstring(occupancy_map_rgb.tobytes(), occupancy_map_rgb.size, occupancy_map_rgb.mode).convert()

        self.screen.fill((0,0,0))
        self.screen.blit(image_surface, (0, 0))
        if occupancy_map_rgb is not None:
            self.screen.blit(occupancy_map_surface, (image_pil.width, 0))

        pygame.display.flip()
        self.clock.tick(60)
        self.last_control = new_control
        return new_control


async def main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    
    carla_world = roar_py_instance.world
    carla_world.set_asynchronous(True)
    carla_world.set_control_steps(0.0, 0.005)
    await carla_world.step()
    roar_py_instance.clean_actors_not_registered()

    # spawn_point, spawn_rpy = carla_world.spawn_points[
    #     np.random.randint(len(carla_world.spawn_points))
    # ]

    spawn_point, spawn_rpy =carla_world.spawn_points[1]
    
    print("Spawning vehicle at", spawn_point, spawn_rpy)

    vehicle = carla_world.spawn_vehicle(
        "vehicle.tesla.model3",
        spawn_point + np.array([0, 0, 2.0]),
        spawn_rpy
    )

    print(vehicle.bounding_box)

    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB, # Specify what kind of data you want to receive
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), # relative position
        np.array([0, 10/180.0*np.pi, 0]), # relative rotation
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