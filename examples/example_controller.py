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

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 6:
            return i % len(waypoints)
    return current_idx

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
    carla_world.set_asynchronous(True)
    carla_world.set_control_steps(0.0, 0.01)
    
    way_points = carla_world.maneuverable_waypoints
    vehicle = carla_world.spawn_vehicle(
        "vehicle.audi.a2",
        way_points[10].location + np.array([0,0,1]),
        way_points[10].roll_pitch_yaw
    )
    current_waypoint_idx = 10
    assert vehicle is not None
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([0.2, 0, 2.0]),
        np.array([0, -5 / 180 * np.pi, 0]),
        120,
        1024,
        768
    )
    assert camera is not None
    try:
        with plt.ion():
            plt.xlim(2000,6000)
            plt.ylim(2000,6000)
            vehicle_location = vehicle.get_3d_location()
            vehicle_rotation = vehicle.get_roll_pitch_yaw()
            vehicle_heading = transforms3d.euler.euler2mat(0,0,vehicle_rotation[2]) @ np.array([1,0,0])
            arrow_heading = plt.arrow(
                vehicle_location[0], 
                vehicle_location[1], 
                vehicle_heading[0] * 100, 
                vehicle_heading[1] * 100, 
                width=50, 
                color='r'
            )
            for waypoint in way_points:
                rep_line = waypoint.line_representation
                rep_line = np.asarray(rep_line)
                plt.plot(rep_line[:,0], rep_line[:,1])
            while True:
                vehicle_location = vehicle.get_3d_location()
                vehicle_rotation = vehicle.get_roll_pitch_yaw()
                vehicle_heading = transforms3d.euler.euler2mat(0,0,vehicle_rotation[2]) @ np.array([1,0,0])
                arrow_heading.set_data(
                    x=vehicle_location[0],
                    y=vehicle_location[1],
                    dx=vehicle_heading[0] * 100,
                    dy=vehicle_heading[1] * 100
                )
                await carla_world.step()
                camera_data = await camera.receive_observation()
                render_ret = manual_viewer.render(camera_data)
                if render_ret is None:
                    break
                current_waypoint_idx = filter_waypoints(
                    vehicle_location,
                    current_waypoint_idx,
                    way_points
                )
                waypoint_to_follow = way_points[current_waypoint_idx + 5]
                vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
                heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])
                delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])
                print(heading_to_waypoint, vehicle_rotation)
                steer_control = (
                    0.5 * delta_heading
                )
                control = {
                    "throttle": 0.2,
                    "steer": steer_control,
                    "brake": 0.0,
                    "hand_brake": 0.0,
                    "reverse": 0,
                    "target_gear": 0
                }
                await vehicle.apply_action(control)
                plt.show()
                plt.pause(0.01)
    finally:
        roar_py_instance.close()


if __name__ == '__main__':
    asyncio.run(main())