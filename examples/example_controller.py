import roar_py_carla_implementation
import roar_py_interface
import carla
import numpy as np
import asyncio
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import pygame
from PIL import Image
import transforms3d

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, waypoints : List[roar_py_interface.RoarPyWaypoint]):
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    list_of_dist = [dist_to_waypoint(w) for w in waypoints]
    min_idx = np.argmin(list_of_dist)
    return waypoints[(min_idx + 3) % len(waypoints)]

class ManualControlViewer:
    def __init__(
        self
    ):
        self.screen = None
        self.clock = None
    
    def init_pygame(self, x, y) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Manual Control Viewer")
        self.clock = pygame.time.Clock()

    def render(self, image : roar_py_interface.RoarPyCameraSensorData) -> Optional[Dict[str, Any]]:
        image_pil : Image = image.get_image()
        if self.screen is None:
            self.init_pygame(image_pil.width, image_pil.height)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        self.screen.fill((0,0,0))
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)
        return {}

async def main():
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla_implementation.RoarPyCarlaInstance(carla_client)
    manual_viewer = ManualControlViewer()
    
    carla_world = roar_py_instance.world
    carla_world.set_control_steps(0.05, 0.005)
    carla_world.set_asynchronous(True)
    
    way_points = carla_world.maneuverable_waypoints
    vehicle = carla_world.spawn_vehicle(
        "vehicle.audi.a2",
        way_points[0].location + np.array([0,0,5]),
        way_points[0].roll_pitch_yaw
    )
    assert vehicle is not None
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([0, 0, 2.0]),
        np.array([0, 0, 0]),
        120,
        1024,
        768
    )
    assert camera is not None
    try:
        while True:
            await carla_world.step()
            camera_data = await camera.receive_observation()
            render_ret = manual_viewer.render(camera_data)
            if render_ret is None:
                break
            waypoint_to_follow = filter_waypoints(
                vehicle.get_3d_location(),
                way_points
            )
            vector_to_waypoint = (waypoint_to_follow.location - vehicle.get_3d_location())[:2]
            heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])
            delta_heading = normalize_rad(heading_to_waypoint - vehicle.get_roll_pitch_yaw()[2])
            print(heading_to_waypoint, vehicle.get_roll_pitch_yaw())
            steer_control = (
                2 * delta_heading
            )
            control = {
                "throttle": 0.4,
                "steer": steer_control,
                "brake": 0.0,
                "hand_brake": 0.0,
                "reverse": 0,
                "target_gear": 0
            }
            await vehicle.apply_action(control)
    finally:
        roar_py_instance.close()


if __name__ == '__main__':
    asyncio.run(main())