import numpy as np
from PIL import Image, ImageDraw, ImageColor
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from serde import serde
from dataclasses import dataclass
from .waypoint import RoarPyWaypoint

@serde
@dataclass
class RoarPyOccupancyMapProducer:
    def __init__(
        self, 
        waypoints : List[RoarPyWaypoint],
        width : int, 
        height : int, 
        width_world : float, 
        height_world : float,
    ):
        self.image = Image.new('L', (width, height), 0)
        self.width_world = width_world
        self.height_world = height_world
        self.waypoints = waypoints

    @property
    def width(self) -> int:
        return self.image.width
    
    @width.setter
    def width(self, value : int):
        self.image = self.image.resize((value, self.height))
    
    @property
    def height(self) -> int:
        return self.image.height
    
    @height.setter
    def height(self, value : int):
        self.image = self.image.resize((self.width, value))

    def  plot_occupancy_map(self, location_2d : np.ndarray, rotation_yaw : float) -> Image:
        assert location_2d.shape == (2, )
        location_min = location_2d - np.array([self.width_world, self.height_world]) # give a bit slack to make sure that we can properly rotate
        location_max = location_2d + np.array([self.width_world, self.height_world])
        
        filtered_waypoint_pairs : List[Tuple[RoarPyWaypoint,RoarPyWaypoint]] = []
        last_waypoint = None
        for i in range(len(self.waypoints) + 1):
            waypoint = self.waypoints[i % len(self.waypoints)]
            if last_waypoint is not None:
                filtered_waypoint_pairs.append((last_waypoint, waypoint))
                last_waypoint = None
            if (
                (np.all(waypoint.line_representation[0][:2] > location_min) and
                np.all(waypoint.line_representation[0][:2] < location_max)) or
                (np.all(waypoint.line_representation[1][:2] > location_min) and
                np.all(waypoint.line_representation[1][:2] < location_max))
            ):
                last_waypoint = waypoint
            
        #self.image = Image.new('L', self.image.size, 0)
        draw = ImageDraw.Draw(self.image)
        for waypoint1, waypoint2 in filtered_waypoint_pairs:
            draw.polygon(
                [
                    self.world_to_pixel(waypoint1.line_representation[0][:2], rotation_yaw, location_2d),
                    self.world_to_pixel(waypoint1.line_representation[1][:2], rotation_yaw, location_2d),
                    self.world_to_pixel(waypoint2.line_representation[1][:2], rotation_yaw, location_2d),
                    self.world_to_pixel(waypoint2.line_representation[0][:2], rotation_yaw, location_2d),
                ],
                fill=255,
            )
        to_ret = self.image
        self.image = Image.new('L', to_ret.size, 0)
        return to_ret
    
    def world_to_pixel(self, location_2d : np.ndarray, rotation_center : float, center_2d : np.ndarray) -> np.ndarray:
        assert location_2d.shape == (2, )
        local_coordinate = location_2d - center_2d
        cos_r = np.cos(rotation_center)
        sin_r = np.sin(rotation_center)

        rotation_matrix = np.array([
            [cos_r, sin_r],
            [-sin_r, cos_r]
        ])
        local_coordinate = rotation_matrix @ local_coordinate

        # Local coordinate in world frame => pixel frame
        local_coordinate = local_coordinate * np.array([self.height / self.height_world, self.width / self.width_world])
        return (-local_coordinate[1] + self.width / 2, -local_coordinate[0] + self.height / 2) 
