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
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
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
            plt.xlim(-20,20)
            plt.ylim(-20,20)
            vehicle_location = vehicle.get_3d_location()
            vehicle_rotation = vehicle.get_roll_pitch_yaw()
            vehicle_heading = transforms3d.euler.euler2mat(0,0,vehicle_rotation[2]) @ np.array([1,0,0])
            arrow_heading = plt.arrow(
                0, 
                0, 
                vehicle_heading[0], 
                vehicle_heading[1], 
                width=0.05, 
                color='r'
            )
            current_waypoint_line = np.asarray(way_points[current_waypoint_idx].line_representation)
            current_waypoint_plt = plt.plot(
                current_waypoint_line[:,0] - vehicle_location[0],
                current_waypoint_line[:,1] - vehicle_location[1]
            )
            current_lookahead_waypoint_line = np.asarray(way_points[(current_waypoint_idx + 3) % len(way_points)].line_representation)
            lookahead_waypoint_plt = plt.plot(
                current_lookahead_waypoint_line[:,0] - vehicle_location[0],
                current_lookahead_waypoint_line[:,1] - vehicle_location[1]
            )
            while True:
                vehicle_location = vehicle.get_3d_location()
                vehicle_rotation = vehicle.get_roll_pitch_yaw()
                vehicle_heading = transforms3d.euler.euler2mat(0,0,vehicle_rotation[2]) @ np.array([1,0,0])
                arrow_heading.set_data(
                    dx=vehicle_heading[0],
                    dy=vehicle_heading[1]
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
                current_waypoint_line = np.asarray(way_points[current_waypoint_idx].line_representation)
                current_waypoint_plt[0].set_data(
                    current_waypoint_line[:,0] - vehicle_location[0],
                    current_waypoint_line[:,1] - vehicle_location[1]
                )
                current_lookahead_waypoint_line = np.asarray(way_points[(current_waypoint_idx + 3) % len(way_points)].line_representation)
                lookahead_waypoint_plt[0].set_data(
                    current_lookahead_waypoint_line[:,0] - vehicle_location[0],
                    current_lookahead_waypoint_line[:,1] - vehicle_location[1]
                )
                waypoint_to_follow = way_points[(current_waypoint_idx + 3) % len(way_points)]
                vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
                heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])
                delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])
                print(heading_to_waypoint, vehicle_rotation)
                steer_control = (
                    -8.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                )
                steer_control = np.clip(steer_control, -1.0, 1.0)
                throttle_control = 0.02 * (40 - np.linalg.norm(vehicle.get_linear_3d_velocity()))

                control = {
                    "throttle": np.clip(throttle_control, 0.0, 1.0),
                    "steer": steer_control,
                    "brake": np.clip(-throttle_control, 0.0, 1.0),
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