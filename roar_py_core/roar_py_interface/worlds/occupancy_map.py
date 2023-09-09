import numpy as np
from PIL import Image, ImageDraw, ImageColor
from typing import List, Optional, Tuple
from serde import serde
from dataclasses import dataclass
from .waypoint import RoarPyWaypoint
import functools
import numba

@numba.jit(nopython=True)
def line_intersects_line(
    line1_start : np.ndarray,
    line1_end : np.ndarray,
    line2_start : np.ndarray,
    line2_end : np.ndarray,
):
    line1_diff = line1_end - line1_start
    line2_diff = line2_end - line2_start
    
    q = (line1_start[1] - line2_start[1]) * (line2_diff[0]) - (line1_start[0] - line2_start[0]) * (line2_diff[1])
    d = (line1_diff[0]) * (line2_diff[1]) - (line1_diff[1]) * (line2_diff[0])

    if d == 0:
        return False

    r = q / d
    q = (line1_start[1] - line2_start[1]) * (line1_diff[0]) - (line1_start[0] - line2_start[0]) * (line1_diff[1])
    s = q / d

    if r < 0 or r > 1 or s < 0 or s > 1:
        return False

    return True

@serde
@dataclass
class RoarPyOccupancyMapProducer:
    """
    The RoarPyOccupancyMapProducer class generates a 2D occupancy map from a list of waypoints.

    The generated map is an image, where each pixel corresponds to a location in the world. The intensity 
    of a pixel represents the occupancy of the corresponding location, with 255 representing an occupied 
    location and 0 representing a free location.

    -----------
    Attributes:
    -----------
        waypoints (List[RoarPyWaypoint]): 
            A list of waypoints that define the lanes.
        width (int): 
            The width of the occupancy map in pixels.
        height (int): 
            The height of the occupancy map in pixels.
        width_world (float): 
            The width of the world in meters. This is used to 
            convert world coordinates to pixel coordinates.
        height_world (float): 
            The height of the world in meters. This is used to 
            convert world coordinates to pixel coordinates.
    """
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
        self.start_index = -1

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

    def plot_occupancy_map(self, location_2d : np.ndarray, rotation_yaw : float) -> Image:
        """
        Generates an occupancy map around a specific location.

        -----
        Args:
        -----
            location_2d (np.ndarray): 
                The 2D coordinates (x, y) of the center of the image in the world frame.
            rotation_yaw (float): 
                The yaw angle (in radians) of the image relative to the world frame. 

        --------
        Returns:
        --------
            Image: 
                An occupancy map centered around the specified location.
        """
        
        assert location_2d.shape == (2, )
        location_min = location_2d - np.array([self.width_world, self.height_world]) # give a bit slack to make sure that we can properly rotate
        location_max = location_2d + np.array([self.width_world, self.height_world])
        

        # Check if the waypoint is in range
        def is_in_range(waypoint : RoarPyWaypoint) -> bool:
            wp_start = waypoint.line_representation[0][:2]
            wp_end = waypoint.line_representation[1][:2]
            segments_to_test = []
            side_1 = np.array([location_min[0], location_max[1]])
            side_2 = np.array([location_max[0], location_min[1]])

            segments_to_test.append((location_min, side_1))
            segments_to_test.append((side_1, location_max))
            segments_to_test.append((location_max, side_2))
            segments_to_test.append((side_2, location_min))

            rst = functools.reduce(
                lambda f, s: f or s,
                [line_intersects_line(wp_start, wp_end, segment[0], segment[1]) for segment in segments_to_test],
                False
            )
            return rst
            # for segment in segments_to_test:
            #     return line_intersects_line(
            #         wp_start[:2], wp_end[:2],
                    
            #     )
            # return (
            #     (np.all(waypoint.line_representation[0][:2] > location_min) and
            #     np.all(waypoint.line_representation[0][:2] < location_max)) or
            #     (np.all(waypoint.line_representation[1][:2] > location_min) and
            #     np.all(waypoint.line_representation[1][:2] < location_max))
            # )

        filtered_waypoint_pairs : List[Tuple[RoarPyWaypoint,RoarPyWaypoint]] = []
        last_waypoint = None

        # Find the starting index of the waypoint in range
        if self.start_index == -1:
            for i in range(len(self.waypoints)):
                waypoint = self.waypoints[i]
                if is_in_range(waypoint):
                    self.start_index = i
                    last_waypoint = waypoint
                    break
        else:
            for i in range(len(self.waypoints)):
                waypoint_idx = (self.start_index - 10 + i) % len(self.waypoints)
                waypoint = self.waypoints[waypoint_idx]
                if is_in_range(waypoint):
                    self.start_index = waypoint_idx
                    last_waypoint = waypoint
                    break

        # Find the waypoint pairs that are in range
        for i in range(1, len(self.waypoints) + 1):
            curr_index = (self.start_index + i) % len(self.waypoints)
            waypoint = self.waypoints[curr_index]

            if last_waypoint is not None:
                filtered_waypoint_pairs.append((last_waypoint, waypoint))
                last_waypoint = None
            
            if is_in_range(waypoint):
                last_waypoint = waypoint
            else:
                break

        # Draw the local occupancy map centered around the actor location
        # self.image = Image.new('L', self.image.size, 0)
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
        """
        Converts a location in the world frame to pixel coordinates in the image frame.

        -----
        Args:
        -----
            location_2d (np.ndarray): 
                The 2D coordinates (x, y) of the location in the world frame.
            rotation_center (float): 
                The yaw angle (in radians) of the image relative to the world frame.
            center_2d (np.ndarray): 
                The 2D coordinates (x, y) of the center of the image in the world frame.

        --------
        Returns:
        --------
            np.ndarray: 
                The pixel coordinates (x, y) of the location in the image frame.
        """
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
