import roar_py_carla
import roar_py_interface
import carla
import numpy as np
import asyncio
from typing import Optional, Dict, Any, List
import pygame
from PIL import Image
import transforms3d
from manual_control import ManualControlViewer

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

# This function filters waypoints using vehicle's location and returns the index of the waypoint closest to the vehicle
def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

async def main():
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    manual_viewer = ManualControlViewer()
    
    carla_world = roar_py_instance.world
    carla_world.set_asynchronous(True)
    carla_world.set_control_steps(0.0, 0.01)
    await carla_world.step()
    roar_py_instance.clean_actors_not_registered()
    
    way_points = carla_world.maneuverable_waypoints
    vehicle = carla_world.spawn_vehicle(
        "vehicle.audi.a2",
        way_points[10].location + np.array([0,0,1]),
        way_points[10].roll_pitch_yaw
    )

    # Initialize current waypoint index to 10 since that's where we spawned the vehicle
    current_waypoint_idx = 10

    assert vehicle is not None
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB, # Specify what kind of data you want to receive
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), # relative position
        np.array([0, 10/180.0*np.pi, 0]), # relative rotation
        image_width=1024,
        image_height=768
    )
    assert camera is not None

    start_time = carla_world.last_tick_elapsed_seconds
    try:
        while True:
            # Step the world first
            await carla_world.step()

            # Get vehicle location and rotation
            vehicle_location = vehicle.get_3d_location()
            vehicle_rotation = vehicle.get_roll_pitch_yaw()

            # Receive camera data and render it
            camera_data = await camera.receive_observation()
            render_ret = manual_viewer.render(camera_data)
            # If user clicked the close button, render_ret will be None
            if render_ret is None:
                break
            
            # Find the waypoint closest to the vehicle
            current_waypoint_idx = filter_waypoints(
                vehicle_location,
                current_waypoint_idx,
                way_points
            )
            # We use the 3rd waypoint ahead of the current waypoint as the target waypoint
            waypoint_to_follow = way_points[(current_waypoint_idx + 3) % len(way_points)]

            # Calculate delta vector towards the target waypoint
            vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
            heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

            # Calculate delta angle towards the target waypoint
            delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

            # Proportional controller to steer the vehicle towards the target waypoint
            steer_control = (
                -8.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
            ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
            steer_control = np.clip(steer_control, -1.0, 1.0)

            # Proportional controller to control the vehicle's speed towards 40 m/s
            throttle_control = 0.05 * (20 - np.linalg.norm(vehicle.get_linear_3d_velocity()))

            control = {
                "throttle": np.clip(throttle_control, 0.0, 1.0),
                "steer": steer_control,
                "brake": np.clip(-throttle_control, 0.0, 1.0),
                "hand_brake": 0.0,
                "reverse": 0,
                "target_gear": 0
            }
            await vehicle.apply_action(control)
    finally:
        roar_py_instance.close()
        end_time = carla_world.last_tick_elapsed_seconds
        delta_time = end_time - start_time
        print(f"Total time: {delta_time} seconds")
        """
        ROAR Competition:
        In ROAR Competition the finished lap count is calculated like what we did above, but there will be logic to verify that you actually finished the lap.
        Also to improve the accuracy the game will be run with fixed time step.
        """


if __name__ == '__main__':
    asyncio.run(main())