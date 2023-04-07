from .sensor import RoarPySensor
from serde import serde
from dataclasses import dataclass
from PIL import Image
import numpy as np
import typing
import gymnasium as gym

class RoarPyCameraSensorData:
    """
    Returns the image as a numpy array.

    """
    def get_image(self) -> Image:
        raise NotImplementedError()
    
    def to_gym(self) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        raise NotImplementedError()

@dataclass
@serde
class RoarPyCameraSensorDataRGB(RoarPyCameraSensorData):
    # RGB image W*H*3, each r/g/b value in range [0,255]
    image_rgb: np.NDArray[np.uint8]

    def get_image(self) -> Image:
        return Image.fromarray(self.image_rgb,mode="RGB")

    def to_gym(self) -> np.ndarray:
        return self.image_rgb

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        return gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)

    @staticmethod
    def from_image(image: Image):
        return __class__(
            np.asarray(image.convert("RGB"), dtype=np.uint8)
        )

@dataclass
@serde
class RoarPyCameraSensorDataGreyscale(RoarPyCameraSensorData):
    # Greyscale image W*H*1, each pixel in range[0,255]
    image_greyscale: np.NDArray[np.uint8]

    def get_image(self) -> Image:
        return Image.fromarray(self.image_greyscale,mode="L")
    
    def to_gym(self) -> np.ndarray:
        return self.image_greyscale

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        return gym.spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    @staticmethod
    def from_image(image: Image):
        return __class__(
            np.asarray(image.convert("L"),dtype=np.uint8)
        )

@dataclass
@serde
class RoarPyCameraSensorDataDepth(RoarPyCameraSensorData):
    # unit in m, W*H*1
    image_depth: np.NDArray[np.float32]
    is_log_scale: bool

    def get_image(self) -> Image:
        # we have to normalize this to [0,1]
        # we should rarely call this function
        min, max = 0, np.max(self.image_depth)
        normalized_image = (self.image_depth) / (max-min)
        return Image.fromarray(normalized_image,mode="F")
    
    def to_gym(self) -> np.ndarray:
        return self.image_depth

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        return gym.spaces.Box(low=0, high=np.inf, shape=(height, width, 1), dtype=np.float32)


@dataclass
@serde
class RoarPyCameraSensorDataSemanticSegmentation(RoarPyCameraSensorData):
    # Semantic Segmentation(SS) Frame, W*H*1
    image_ss: np.NDArray[np.uint64]
    # Dictionary mapping each pixel in SS Frame to a RGB array of color and a label
    ss_label_color_map: dict[int,tuple[np.NDArray[np.uint8], str]]

    def get_image(self) -> Image:
        # we have to normalize this to [0,1]
        # we should rarely call this function
        image_RGB = np.zeros(shape=(*self.image_ss.shape[:-1],3),dtype=np.uint8)
        for w in self.image_ss.shape[0]:
            for h in self.image_ss.shape[1]:
                image_RGB[w,h,:] = self.ss_label_color_map[
                    self.image_ss[w,h,0]
                ]
        
        return Image.fromarray(image_RGB,mode="RGB")
    
    def to_gym(self) -> np.ndarray:
        return self.image_ss

    @staticmethod
    def gym_observation_space(width : int, height : int) -> gym.Space:
        return gym.spaces.Box(low=0, high=np.inf, shape=(height, width, 1), dtype=np.uint64)

class RoarPyCameraSensor(RoarPySensor[RoarPyCameraSensorData]):
    def convert_obs_to_gym_obs(self, obs: RoarPyCameraSensorData):
        return obs.to_gym()

