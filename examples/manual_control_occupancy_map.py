import roar_py_carla
import roar_py_interface
import carla
import numpy as np
import asyncio
from manual_control import ManualControlViewer

async def main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(15.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)

    carla_world = roar_py_instance.world
    carla_world.set_asynchronous(True)
    carla_world.set_control_steps(0.0, 0.005)
    await carla_world.step()
    roar_py_instance.clean_actors_not_registered()

    print("Map Name", carla_world.map_name)
    occ_map_producer = roar_py_interface.RoarPyOccupancyMapProducer(carla_world.maneuverable_waypoints, 100, 100, 5, 5)
    
    spawn_point, spawn_rpy =carla_world.spawn_points[0]
    
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
            occupancy_map = occ_map_producer.plot_occupancy_map(vehicle.get_3d_location()[:2], vehicle.get_roll_pitch_yaw()[2])
            control = viewer.render(img, occupancy_map)
            if control is None:
                break
            await vehicle.apply_action(control)
    finally:
        roar_py_instance.close()
    

if __name__ == '__main__':
    asyncio.run(main())
