import numpy as np
from typing import Tuple, List
import transforms3d as tr3d
from serde import serde
from dataclasses import dataclass
from functools import cached_property

def normalize_rad(radians : float) -> float:
    return (radians + np.pi) % (2 * np.pi) - np.pi

@serde
@dataclass
class RoarPyWaypoint:
    location: np.ndarray        # x, y, z of the center of the waypoint
    roll_pitch_yaw: np.ndarray  # rpy of the road in radians, note that a road in the forward direction of the robot means their rpys are the same
    lane_width: float           # width of the lane in meters at this waypoint
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, RoarPyWaypoint):
            return False
        return np.allclose(self.location, __value.location) and np.allclose(self.roll_pitch_yaw, __value.roll_pitch_yaw) and np.allclose(self.lane_width, __value.lane_width)

    @cached_property
    def line_representation(self) -> Tuple[np.ndarray, np.ndarray]:
        mid_point = self.location
        local_coordinate_pos = np.array([0, self.lane_width/2, 0])
        local_coordinate_neg = -local_coordinate_pos
        rotation_matrix = tr3d.euler.euler2mat(*self.roll_pitch_yaw)
        global_coordinate_pos = mid_point + rotation_matrix.dot(local_coordinate_pos)
        global_coordinate_neg = mid_point + rotation_matrix.dot(local_coordinate_neg)
        
        return global_coordinate_pos, global_coordinate_neg

    @staticmethod
    def load_waypoint_list(waypoint_dict : dict) -> List['RoarPyWaypoint']:
        flattened_lane_widths = waypoint_dict['lane_widths'].flatten()
        return [RoarPyWaypoint(
            waypoint_dict['locations'][i],
            waypoint_dict['rotations'][i],
            flattened_lane_widths[i]
        ) for i in range(len(waypoint_dict['locations']))]

    @staticmethod
    def save_waypoint_list(waypoints: List['RoarPyWaypoint']) -> dict:
        return {
            'locations': np.stack([waypoint.location for waypoint in waypoints], axis=0),
            'rotations': np.stack([waypoint.roll_pitch_yaw for waypoint in waypoints], axis=0),
            'lane_widths': np.stack([waypoint.lane_width for waypoint in waypoints], axis=0)
        }
    
    @staticmethod
    def from_line_representation(
        point_1: np.ndarray,
        point_2: np.ndarray,
        roll_pitch_yaw: np.ndarray,
    ) -> "RoarPyWaypoint":
        midpoint = (point_1 + point_2) / 2
        line_length = np.linalg.norm(point_1 - point_2)
        return RoarPyWaypoint(
            midpoint,
            roll_pitch_yaw,
            line_length
        )

    @staticmethod
    def interpolate(point_1 : "RoarPyWaypoint", point_2 : "RoarPyWaypoint", alpha : float) -> "RoarPyWaypoint":
        location = point_1.location * alpha + point_2.location * (1-alpha)
        roll_pitch_yaw = normalize_rad(point_1.roll_pitch_yaw * alpha + point_2.roll_pitch_yaw * (1-alpha))
        lane_width = point_1.lane_width * alpha + point_2.lane_width * (1-alpha)
        return RoarPyWaypoint(location, roll_pitch_yaw, lane_width)
