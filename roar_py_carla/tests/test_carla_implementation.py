from roar_py_carla import RoarPyCarlaInstance, RoarPyCarlaVehicle
import roar_py_interface
import carla
import pytest
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

@pytest.fixture(scope="session")
def carla_instance() -> RoarPyCarlaInstance:
    """
    This function initializes a connection to a CARLA server and returns a RoarPyCarlaInstance object.
    """
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = RoarPyCarlaInstance(carla_client)
    
    yield roar_py_instance

    roar_py_instance.close()

@pytest.fixture(scope="session")
def carla_vehicle(carla_instance: RoarPyCarlaInstance) -> RoarPyCarlaVehicle:
    """
    This function spawns a vehicle and returns a RoarPyCarlaVehicle object.
    """
    vehicle_blueprint_id = "vehicle.tesla.model3"
    spawn_point = carla_instance.world.spawn_points[np.random.randint(len(carla_instance.world.spawn_points))]

    vehicle = carla_instance.world.spawn_vehicle(
        vehicle_blueprint_id,
        spawn_point[0],
        spawn_point[1]
    )
    
    assert vehicle is not None, "Failed to spawn vehicle"

    yield vehicle

    vehicle.close()

@pytest.mark.parametrize("is_async", [
    True,
    False
])
@pytest.mark.parametrize("target_data_type", [
    roar_py_interface.RoarPyCameraSensorDataRGB,
    roar_py_interface.RoarPyCameraSensorDataGreyscale,
    roar_py_interface.RoarPyCameraSensorDataDepth,
    roar_py_interface.RoarPyCameraSensorDataSemanticSegmentation
])
@pytest.mark.parametrize("image_size", [
    (64, 64),
    (256, 256)
])
@pytest.mark.parametrize("fov", [
    90
])
@pytest.mark.asyncio
async def test_camera(
    carla_instance : RoarPyCarlaInstance, 
    carla_vehicle : RoarPyCarlaVehicle,
    is_async : bool,
    target_data_type : roar_py_interface.RoarPyCameraSensorData,
    image_size : Tuple[int, int],
    fov: float,
    num_frames: int = 10 * 10
):
    carla_instance.world.set_asynchronous(is_async)
    carla_instance.world.set_control_steps(0.1, 0.05)

    camera = carla_vehicle.attach_camera_sensor(
        target_data_type,
        np.array([0, 0, 2.5]),
        np.array([0, 0, 0]),
        fov,
        image_size[0],
        image_size[1]
    )
    
    rendered_frames = []

    for _ in range(num_frames):
        await carla_instance.world.step()
        img : roar_py_interface.RoarPyCameraSensorData = await camera.receive_observation()
        img_pil = img.get_image()
        assert img_pil.size == image_size
        # img.get_image().save("test.png")
        rendered_frames.append(img_pil)
    
    assert len(rendered_frames) == num_frames
    camera.close()

@pytest.mark.parametrize("is_async", [
    True,
    False
])
@pytest.mark.asyncio
async def test_collision_sensor(
    carla_instance : RoarPyCarlaInstance, 
    carla_vehicle : RoarPyCarlaVehicle, 
    is_async : bool
):
    carla_instance.world.set_asynchronous(is_async)
    carla_instance.world.set_control_steps(0.1, 0.05)

    assert carla_vehicle is not None

    collision_sensor = carla_vehicle.attach_collision_sensor(
        np.array([0, 0, 0]),
        np.array([0, 0, 0])
    )

    assert collision_sensor is not None
    collision_sensor.close()

@pytest.mark.parametrize("is_async", [
    True,
    False
])
@pytest.mark.parametrize("num_lasers", [
    12,
    24
])
@pytest.mark.parametrize("max_distance", [
    10.0
])
@pytest.mark.parametrize("points_per_second", [
    10000
])
@pytest.mark.parametrize("rotation_frequency", [
    10.0
])
@pytest.mark.parametrize("upper_fov", [
    10.0
])
@pytest.mark.parametrize("lower_fov", [
    -30.0
])
@pytest.mark.parametrize("horizontal_fov", [
    120.0,
    360.0
])
@pytest.mark.parametrize("atmosphere_attenuation_rate", [
    0.004
])
@pytest.mark.asyncio
async def test_lidar_sensor(
    carla_instance : RoarPyCarlaInstance, 
    carla_vehicle : RoarPyCarlaVehicle,
    is_async : bool,
    num_lasers : int,
    max_distance : float,
    points_per_second : int,
    rotation_frequency : float,
    upper_fov : float,
    lower_fov : float,
    horizontal_fov : float,
    atmosphere_attenuation_rate : float
):
    carla_instance.world.set_asynchronous(is_async)
    carla_instance.world.set_control_steps(0.1, 0.05)

    lidar_sensor = carla_vehicle.attach_lidar_sensor(
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        num_lasers=num_lasers,
        max_distance=max_distance,
        points_per_second=points_per_second,
        rotation_frequency=rotation_frequency,
        upper_fov=upper_fov,
        lower_fov=lower_fov,
        horizontal_fov=horizontal_fov,
        atmosphere_attenuation_rate=atmosphere_attenuation_rate
    )

    assert lidar_sensor is not None

    lidar_sensor.close()

@pytest.mark.parametrize("is_async", [
    True,
    False
])
@pytest.mark.asyncio
async def test_vehicle_control(
    carla_instance : RoarPyCarlaInstance,
    carla_vehicle : RoarPyCarlaVehicle,
    is_async : bool
):
    carla_instance.world.set_asynchronous(is_async)
    carla_instance.world.set_control_steps(0.1, 0.05)
    obs_spec = carla_vehicle.get_gym_observation_spec()
    act_spec = carla_vehicle.get_action_spec()
    for _ in range(10*10):
        await carla_vehicle.receive_observation()
        obs = carla_vehicle.get_last_gym_observation()
        control = act_spec.sample()
        assert (await carla_vehicle.apply_action(control)) == True
        await carla_instance.world.step()

@pytest.mark.parametrize("is_async", [
    True,
    False
])
@pytest.mark.asyncio
async def test_map_manuverable_waypoints(
    carla_instance : RoarPyCarlaInstance,
    carla_vehicle : RoarPyCarlaVehicle,
    is_async : bool
):
    carla_instance.world.set_asynchronous(is_async)
    carla_instance.world.set_control_steps(0.1, 0.05)
    all_waypoints = carla_instance.world.maneuverable_waypoints
    assert len(all_waypoints) > 0