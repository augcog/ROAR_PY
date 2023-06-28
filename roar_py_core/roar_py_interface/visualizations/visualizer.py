from ..actors import RoarPyActor
from ..sensors import RoarPyCameraSensorData, RoarPyOccupancyMapSensorData
from PIL import Image
from typing import Optional

class RoarPyVisualizer:
    def __init__(self, actor : RoarPyActor) -> None:
        self.actor = actor

    def render(self) -> Optional[Image.Image]:
        last_dat = self.actor.get_last_observation()
        if last_dat is None:
            return None

        max_width = 0
        max_height = 0
        for key, dat in last_dat.items():
            if isinstance(dat, RoarPyCameraSensorData):
                size = dat.get_size()
            elif isinstance(dat, RoarPyOccupancyMapSensorData):
                size = dat.occupancy_map.size
            else:
                continue
            max_width = max(max_width, size[0])
            max_height = max(max_height, size[1])
        
        if max_width == 0 or max_height == 0:
            return None
        
        image = Image.new("RGB", (max_width, max_height))
        for key, dat in last_dat.items():
            if isinstance(dat, RoarPyCameraSensorData):
                image.paste(dat.get_image(), (0, 0))
            elif isinstance(dat, RoarPyOccupancyMapSensorData):
                image.paste(dat.occupancy_map, (max_width - dat.occupancy_map.width, 0))
            else:
                continue
        return image