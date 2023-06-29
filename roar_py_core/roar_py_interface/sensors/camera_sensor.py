from roar_py_interface.base.sensor import RoarPyRemoteSupportedSensorSerializationScheme
from ..base import RoarPySensor, RoarPyRemoteSupportedSensorData, RoarPyRemoteSupportedSensorSerializationScheme
from ..base.sensor import remote_support_sensor_data_register
from serde import serde
from dataclasses import dataclass
from PIL import Image
import numpy as np
import typing
import gymnasium as gym
import io

class RoarPyCameraSensorData(RoarPyRemoteSupportedSensorData):
    """
    Returns the image as a numpy array.

    """
    def get_image(self) -> Image:
        raise NotImplementedError()
    
    def to_gym(self) -> typing.Any:
        raise NotImplementedError()
    
    def get_size(self) -> typing.Tuple[int, int]:
        return self.get_image().size

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        raise NotImplementedError()
    
    def get_gym_observation_spec(self) -> gym.Space:
        return self.__class__.gym_observation_space(*self.get_size())

    def convert_obs_to_gym_obs(self):
        return self.to_gym()

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyCameraSensorDataRGB(RoarPyCameraSensorData):
    # RGB image W*H*3, each r/g/b value in range [0,255]
    image_rgb: np.ndarray #np.NDArray[np.uint8]

    def get_image(self) -> Image:
        return Image.fromarray(self.image_rgb,mode="RGB")

    def to_gym(self) -> np.ndarray:
        return self.image_rgb
    
    def get_size(self) -> typing.Tuple[int, int]:
        return self.image_rgb.shape[1::-1]

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        return gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)

    @staticmethod
    def from_image(image: Image):
        return __class__(
            np.asarray(image.convert("RGB"), dtype=np.uint8)
        )

    def to_data(self, scheme: RoarPyRemoteSupportedSensorSerializationScheme) -> bytes:
        saved_image = io.BytesIO()
        self.get_image().save(saved_image, format="JPEG")
        return saved_image.getvalue()

    @staticmethod
    def from_data_custom(data : bytes, scheme : RoarPyRemoteSupportedSensorSerializationScheme):
        image_bytes = io.BytesIO(data)
        image_bytes.seek(0)
        img = Image.open(image_bytes)
        return __class__.from_image(img)


@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyCameraSensorDataGreyscale(RoarPyCameraSensorData):
    # Greyscale image W*H*1, each pixel in range[0,255]
    image_greyscale: np.ndarray #np.NDArray[np.uint8]

    def get_image(self) -> Image:
        return Image.fromarray(self.image_greyscale,mode="L")
    
    def to_gym(self) -> np.ndarray:
        return self.image_greyscale
    
    def get_size(self) -> typing.Tuple[int, int]:
        return self.image_greyscale.shape[1::-1]

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        return gym.spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    @staticmethod
    def from_image(image: Image):
        return __class__(
            np.asarray(image.convert("L"),dtype=np.uint8)
        )

    def to_data(self, scheme: RoarPyRemoteSupportedSensorSerializationScheme) -> bytes:
        saved_image = io.BytesIO()
        self.get_image().save(saved_image, format="JPEG")
        return saved_image.getvalue()

    @staticmethod
    def from_data_custom(data : bytes, scheme : RoarPyRemoteSupportedSensorSerializationScheme):
        image_bytes = io.BytesIO(data)
        image_bytes.seek(0)
        img = Image.open(image_bytes)
        return __class__.from_image(img)

@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyCameraSensorDataDepth(RoarPyCameraSensorData):
    # unit in m, W*H*1
    image_depth: np.ndarray #np.NDArray[np.float32]
    is_log_scale: bool

    def get_image(self) -> Image:
        # we have to normalize this to [0,1]
        # we should rarely call this function
        min, max = np.min(self.image_depth), np.max(self.image_depth)
        normalized_image = (self.image_depth) / (max-min)
        normalized_image = (normalized_image * 255).astype(np.uint8)
        return Image.fromarray(normalized_image,mode="L")
    
    def to_gym(self) -> np.ndarray:
        return self.image_depth
    
    def get_size(self) -> typing.Tuple[int, int]:
        return self.image_depth.shape[1::-1]

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        return gym.spaces.Box(low=0, high=np.inf, shape=(height, width, 1), dtype=np.float32)


@remote_support_sensor_data_register
@serde
@dataclass
class RoarPyCameraSensorDataSemanticSegmentation(RoarPyCameraSensorData):
    # Semantic Segmentation(SS) Frame, W*H*1
    image_ss: np.ndarray #np.NDArray[np.uint64]
    # Dictionary mapping each pixel in SS Frame to a RGB array of color and a label
    ss_label_color_map: typing.Dict[int,typing.Tuple[np.ndarray, str]] #typing.Dict[int,typing.Tuple[np.NDArray[np.uint8], str]]

    def get_image(self) -> Image:
        # we have to normalize this to [0,1]
        # we should rarely call this function
        image_RGB = np.zeros(shape=(*self.image_ss.shape[:-1],3),dtype=np.uint8)
        for w in range(self.image_ss.shape[0]):
            for h in range(self.image_ss.shape[1]):
                image_RGB[w,h,:] = self.ss_label_color_map[
                    self.image_ss[w,h,0]
                ][0]
        
        return Image.fromarray(image_RGB,mode="RGB")
    
    def to_gym(self) -> np.ndarray:
        return self.image_ss
    
    def get_size(self) -> typing.Tuple[int, int]:
        return self.image_ss.shape[1::-1]

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        return gym.spaces.Box(low=0, high=np.iinfo(np.uint64).max, shape=(height, width, 1), dtype=np.uint64)

class RoarPyCameraSensor(RoarPySensor[RoarPyCameraSensorData]):
    sensordata_type = RoarPyCameraSensorData
    def __init__(
        self,
        name: str,
        control_timestep: float,
        target_data_type: typing.Type[RoarPyCameraSensorData]
    ):
        super().__init__(name, control_timestep)
        self.sensordata_type = target_data_type

    @property
    def image_size_width(self) -> int:
        raise NotImplementedError()
    
    @property
    def image_size_height(self) -> int:
        raise NotImplementedError()
    
    @property
    def fov(self) -> float:
        raise NotImplementedError()
    
    def get_gym_observation_spec(self) -> gym.Space:
        return self.sensordata_type.gym_observation_space(self.image_size_width, self.image_size_height)

    def convert_obs_to_gym_obs(self, obs: RoarPyCameraSensorData):
        return obs.to_gym()

