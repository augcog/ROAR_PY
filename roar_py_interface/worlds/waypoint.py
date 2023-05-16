import numpy as np

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