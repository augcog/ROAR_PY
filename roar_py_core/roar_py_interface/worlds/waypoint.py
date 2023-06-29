import numpy as np
from typing import Tuple, List
import transforms3d as tr3d
from serde import serde
from dataclasses import dataclass
from functools import cached_property

@serde
@dataclass
class RoarPyWaypoint:
    location: np.ndarray        # x, y, z of the center of the waypoint
    roll_pitch_yaw: np.ndarray  # rpy of the road in radians, note that a road in the forward direction of the robot means their rpys are the same
    lane_width: float           # width of the lane in meters at this waypoint
    
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
    