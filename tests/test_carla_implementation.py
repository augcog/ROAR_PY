from roar_py_carla_implementation import RoarPyCarlaInstance
import roar_py_interface
import carla
import pytest
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

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
    (256, 256),
    (800, 600)
])
@pytest.mark.parametrize("fov", [
    90,
    120
])
@pytest.mark.asyncio
async def test_camera(
    carla_instance : RoarPyCarlaInstance, 
    is_async : bool,
    target_data_type : roar_py_interface.RoarPyCameraSensorData,
    image_size : Tuple[int, int],
    fov: float,
    num_frames: int = 10 * 10
):
    carla_instance.world.set_asynchronous(is_async)
    carla_instance.world.set_control_steps(0.1, 0.05)
    spawn_point = carla_instance.world.spawn_points[np.random.randint(len(carla_instance.world.spawn_points))]

    vehicle = carla_instance.world.spawn_vehicle(
        "vehicle.tesla.model3",
        spawn_point[0],
        spawn_point[1]
    )

    assert vehicle is not None

    camera = vehicle.attach_camera_sensor(
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
    vehicle.close()
