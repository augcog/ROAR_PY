import numpy as np
from typing import Tuple
import transforms3d as tr3d
from functools import cached_property

class RoarPyWaypoint:
    def __init__(
        self,
        location: np.ndarray,
        roll_pitch_yaw: np.ndarray, # rpy of the road in radians, note that a road in the forward direction of the robot means their rpys are the same
        lane_width: float # in meters
    ):
        self.location = location
        self.roll_pitch_yaw = roll_pitch_yaw
        self.lane_width = lane_width
    
    @cached_property
    def line_representation(self) -> Tuple[np.ndarray, np.ndarray]:
        mid_point = self.location
        local_coordinate_pos = np.array([0, self.lane_width/2, 0])
        local_coordinate_neg = -local_coordinate_pos
        rotation_matrix = tr3d.euler.euler2mat(*self.roll_pitch_yaw)
        global_coordinate_pos = mid_point + rotation_matrix.dot(local_coordinate_pos)
        global_coordinate_neg = mid_point + rotation_matrix.dot(local_coordinate_neg)
        
        return global_coordinate_pos, global_coordinate_neg