from roar_py_interface import RoarPyCameraSensor, RoarPyCameraSensorDataRGB, RoarPyCameraSensorDataDepth, RoarPyCameraSensorDataGreyscale, RoarPyCameraSensorDataSemanticSegmentation, RoarPyCameraSensorData, roar_py_thread_sync
import typing
import gymnasium as gym
import carla
import asyncio
import numpy as np
from PIL import Image
from ..base import RoarPyCarlaBase

def __convert_carla_image_to_bgra_array(
    carla_data: carla.Image,
    width: int,
    height: int
) -> np.ndarray: #np.NDArray[np.uint8]:
    
    array_dat = np.frombuffer(carla_data.raw_data, dtype=np.uint8)
    array_dat = np.reshape(array_dat, (height, width, 4))
    return array_dat

def __depth_meters_from_carla_bgra(
    bgra_carla: np.ndarray
) -> np.ndarray: #np.NDArray[np.float32]:
    assert bgra_carla.ndim == 3
    assert bgra_carla.shape[-1] >= 3
    #bgra_carla = bgra_carla.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) <= This would be in mm
    normalized_depth = np.dot(bgra_carla[:, :, :3], [65536.0, 256.0, 1.0])
    # We omit the following normalizing step in carla
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    # Then we transform the depth in meters
    normalized_depth *= 1000.0
    return normalized_depth

#https://carla.readthedocs.io/en/0.9.14/ref_sensors/#semantic-segmentation-camera
#Valid version: 0.9.12 - 0.9.14
__carla_semantic_segmentation_color_map = {
    0: (np.array([0, 0, 0],dtype=np.uint8),"Unlabeled"),
    1: (np.array([70,70,70],dtype=np.uint8) ,"Building"),
    2: (np.array([100,40,40],dtype=np.uint8),"Fence"),
    3: (np.array([55,90,80],dtype=np.uint8),"Other"),
    4: (np.array([220,20,60],dtype=np.uint8),"Pedestrian"),
    5: (np.array([153,153,153],dtype=np.uint8),"Pole"),
    6: (np.array([157,234,50],dtype=np.uint8),"RoadLine"),
    7: (np.array([128,64,128],dtype=np.uint8),"Road"),
    8: (np.array([244,35,232],dtype=np.uint8),"Sidewalk"),
    9: (np.array([107,142,35],dtype=np.uint8),"Vegetation"),
    10: (np.array([0,0,142],dtype=np.uint8),"Car"),
    11: (np.array([102,102,156],dtype=np.uint8),"Wall"),
    12: (np.array([220,220,0],dtype=np.uint8),"TrafficSign"),
    13: (np.array([70,130,180],dtype=np.uint8),"Sky"),
    14: (np.array([81,0,81],dtype=np.uint8),"Ground"),
    15: (np.array([150,100,100],dtype=np.uint8),"Bridge"),
    16: (np.array([230,150,140],dtype=np.uint8),"RailTrack"),
    17: (np.array([180,165,180],dtype=np.uint8),"GuardRail"),
    18: (np.array([250,170,30],dtype=np.uint8),"TrafficLight"),
    19: (np.array([110,190,160],dtype=np.uint8),"Static"),
    20: (np.array([170,120,50],dtype=np.uint8),"Dynamic"),
    21: (np.array([45,60,150],dtype=np.uint8),"Water"),
    22: (np.array([145,170,100],dtype=np.uint8),"Terrain")
}

def _convert_carla_to_roarpy_image(blueprint_id : str, width : int, height : int, target_data_type : typing.Type[RoarPyCameraSensorData], carla_data: carla.Image) -> RoarPyCameraSensorData:
    #https://github.com/carla-simulator/carla/blob/master/LibCarla/source/carla/image/ColorConverter.h
    #https://github.com/carla-simulator/data-collector/blob/master/carla/image_converter.py

    assert blueprint_id in ["sensor.camera.depth", "sensor.camera.semantic_segmentation", "sensor.camera.rgb", "sensor.camera.instance_segmentation"], "Unsupported blueprint_id: {} for carla camera sensor support".format(blueprint_id)

    ret_image_bgra = __convert_carla_image_to_bgra_array(carla_data, width, height)
    if target_data_type == RoarPyCameraSensorDataRGB:
        assert blueprint_id == "sensor.camera.rgb", "Cannot convert {} to RoarPyCameraSensorDataRGB".format(blueprint_id)
        return RoarPyCameraSensorDataRGB(
            ret_image_bgra[:,:,2::-1] #[:,:,:3][:,:,::-1]
        )
    elif target_data_type == RoarPyCameraSensorDataGreyscale:
        assert blueprint_id == "sensor.camera.rgb", "Cannot convert {} to RoarPyCameraSensorDataGreyscale".format(blueprint_id)
        img = Image.fromarray(ret_image_bgra[:,:,:3][:,:,::-1],mode="RGB")
        grey_img = img.convert("L")
        img.close()
        ret = RoarPyCameraSensorDataGreyscale(
            np.asarray(grey_img, dtype=np.uint8)
        )
        grey_img.close()
        return ret
    elif target_data_type == RoarPyCameraSensorDataDepth:
        assert blueprint_id == "sensor.camera.depth", "Cannot convert {} to RoarPyCameraSensorDataDepth".format(blueprint_id)
        return RoarPyCameraSensorDataDepth(
            __depth_meters_from_carla_bgra(ret_image_bgra),
            is_log_scale=False
        )
    elif target_data_type == RoarPyCameraSensorDataSemanticSegmentation:
        assert blueprint_id == "sensor.camera.semantic_segmentation" or blueprint_id == "sensor.camera.instance_segmentation", "Cannot convert {} to RoarPyCameraSensorDataSemanticSegmentation".format(blueprint_id)
        # label encoded in the red channel: A pixel with a red value of x belongs to an object with tag x
        # for instance segmentation sensor, The green and blue values of the pixel define the object's unique ID. 
        # Code cross-checked with https://github.com/carla-simulator/data-collector/blob/master/carla/image_converter.py, should be ok
        ret_labels = ret_image_bgra[:,:,2:3].astype(np.uint64)

        return RoarPyCameraSensorDataSemanticSegmentation(
            ret_labels,
            __carla_semantic_segmentation_color_map
        )
    else:
        raise NotImplementedError("Unsupported target_data_type: {}".format(target_data_type))


class RoarPyCarlaCameraSensor(RoarPyCameraSensor,RoarPyCarlaBase):
    SUPPORTED_BLUEPRINT_TO_TARGET_DATA = {
        "sensor.camera.rgb": [RoarPyCameraSensorDataRGB, RoarPyCameraSensorDataGreyscale],
        "sensor.camera.depth": [RoarPyCameraSensorDataDepth],
        "sensor.camera.semantic_segmentation": [RoarPyCameraSensorDataSemanticSegmentation],
        "sensor.camera.instance_segmentation": [RoarPyCameraSensorDataSemanticSegmentation]
    }
    SUPPORTED_TARGET_DATA_TO_BLUEPRINT = {
        RoarPyCameraSensorDataRGB: "sensor.camera.rgb",
        RoarPyCameraSensorDataGreyscale: "sensor.camera.rgb",
        RoarPyCameraSensorDataDepth: "sensor.camera.depth",
        RoarPyCameraSensorDataSemanticSegmentation: "sensor.camera.semantic_segmentation"
    }
    def __init__(
        self, 
        carla_instance: "RoarPyCarlaInstance",
        sensor: carla.Sensor,
        target_data_type: typing.Optional[typing.Type[RoarPyCameraSensorData]] = None,
        name: str = "carla_camera",
    ):
        RoarPyCarlaBase.__init__(self, carla_instance, sensor)
        assert sensor.type_id in __class__.SUPPORTED_BLUEPRINT_TO_TARGET_DATA.keys(), "Unsupported blueprint_id: {} for carla camera sensor support".format(sensor.type_id)
        if target_data_type is None:
            target_data_type = __class__.SUPPORTED_BLUEPRINT_TO_TARGET_DATA[sensor.type_id][0]
        assert target_data_type in __class__.SUPPORTED_BLUEPRINT_TO_TARGET_DATA[sensor.type_id], "Unsupported target_data_type: {} for blueprint_id: {}".format(target_data_type, sensor.type_id)

        RoarPyCameraSensor.__init__(self, name = name, control_timestep = 0.0, target_data_type = target_data_type)
        self.sensordata_type = target_data_type
        sensor.listen(
            self.listen_carla_data
        )
        self.received_data : typing.Optional[RoarPyCameraSensorData] = None

    @property
    def control_timestep(self) -> float:
        return float(self._base_actor.attributes["sensor_tick"])

    @property
    def fov(self) -> float:
        return float(self._base_actor.attributes["fov"])

    @property
    def image_size_width(self) -> int:
        return int(self._base_actor.attributes["image_size_x"])

    @property
    def image_size_height(self) -> int:
        return int(self._base_actor.attributes["image_size_y"])
    
    def listen_carla_data(self, carla_data: carla.Image) -> None:
        self.received_data = _convert_carla_to_roarpy_image(
            self._base_actor.type_id,
            self.image_size_width,
            self.image_size_height,
            self.sensordata_type,
            carla_data
        )

    def get_gym_observation_spec(self) -> gym.Space:
        return self.sensordata_type.gym_observation_space(self.image_size_width, self.image_size_height)
    
    async def receive_observation(self) -> RoarPyCameraSensorData:
        while self.received_data is None:
            await asyncio.sleep(0.001)
        return self.received_data
    
    def get_last_observation(self) -> typing.Optional[RoarPyCameraSensorData]:
        return self.received_data
    
    @roar_py_thread_sync
    def close(self):
        if self._base_actor is not None and self._base_actor.is_listening:
            self._base_actor.stop()
        RoarPyCarlaBase.close(self)
    
    @roar_py_thread_sync
    def is_closed(self) -> bool:
        return self._base_actor is None or not self._base_actor.is_listening