import numpy as np
from typing import Tuple, List, Optional, Union
import transforms3d as tr3d
from serde import serde
from dataclasses import dataclass
from functools import cached_property
from shapely import Polygon, Point
from collections import namedtuple
import math
import copy

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

    @staticmethod
    def distance_to_waypoint_polygon(
        waypoint1: "RoarPyWaypoint",
        waypoint2: "RoarPyWaypoint",
        point: Union[np.ndarray, Point]
    ):
        p1, p2 = waypoint1.line_representation
        p3, p4 = waypoint2.line_representation
        polygon = Polygon([p1[:2], p2[:2], p4[:2], p3[:2]])
        if isinstance(point, np.ndarray):
            return polygon.distance(Point(point[:2]))
        else:
            return polygon.distance(point)

RoarPyWaypointsProjection = namedtuple("RoarPyWaypointsProjectResult", ["waypoint_idx", "distance_from_waypoint"])

class RoarPyWaypointsTracker:
    waypoints: List[RoarPyWaypoint]

    def __init__(
        self,
        waypoints: List[RoarPyWaypoint],
        current_traced_index: int = 0
    ):
        assert len(waypoints) > 1
        self._waypoints = waypoints
        self._distance_between_waypoints : List[float] = []
        self._total_distance_from_first_waypoint : List[float] = []
        self._total_distance = 0
        self._rebuild_waypoints_distances()
        self.current_traced_index = current_traced_index

    @property
    def waypoints(self) -> List[RoarPyWaypoint]:
        return self._waypoints
    
    @waypoints.setter
    def waypoints(self, waypoints: List[RoarPyWaypoint]):
        assert len(waypoints) > 1
        self._waypoints = waypoints
        self._rebuild_waypoints_distances()
    
    def _rebuild_waypoints_distances(self) -> None:
        self._distance_between_waypoints = []
        self._total_distance_from_first_waypoint = []
        self._total_distance = 0
        for i in range(len(self._waypoints)):
            self._total_distance_from_first_waypoint.append(self._total_distance)
            if i == len(self._waypoints) - 1:
                self._distance_between_waypoints.append(np.linalg.norm(self._waypoints[i].location - self._waypoints[0].location))
            else:
                self._distance_between_waypoints.append(np.linalg.norm(self._waypoints[i].location - self._waypoints[i+1].location))
            self._total_distance += self._distance_between_waypoints[-1]

    def trace_point(self, point: np.ndarray, start_idx : int = 0) -> RoarPyWaypointsProjection:
        """
        Trace a point to the closest waypoint
        :param point: point to be traced
        :param start_idx: index to start tracing
        :return: index of the closest waypoint
        """
        size_of_waypoints = len(self.waypoints)
        assert size_of_waypoints > 1

        point_p = Point(point[:2])

        min_dist_idx, min_dist = 0, float("inf")
        for i in range(1, int(math.ceil(size_of_waypoints/2)) + 2):
            forward_waypoint_prev = self.waypoints[(start_idx + i - 1) % size_of_waypoints]
            forward_waypoint_after = self.waypoints[(start_idx + i) % size_of_waypoints]
            backward_waypoint_prev = self.waypoints[(start_idx - i) % size_of_waypoints]
            backward_waypoint_after = self.waypoints[(start_idx - i + 1) % size_of_waypoints]
            forward_distance = RoarPyWaypoint.distance_to_waypoint_polygon(forward_waypoint_prev, forward_waypoint_after, point_p)
            backward_distance = RoarPyWaypoint.distance_to_waypoint_polygon(backward_waypoint_prev, backward_waypoint_after, point_p)
            if forward_distance < min_dist:
                min_dist_idx = (start_idx + i - 1) % size_of_waypoints
                min_dist = forward_distance
            if backward_distance < min_dist:
                min_dist_idx = (start_idx - i) % size_of_waypoints
                min_dist = backward_distance
            if min_dist == 0:
                break
        
        prev_waypoint = self.waypoints[min_dist_idx]
        after_waypoint = self.waypoints[(min_dist_idx + 1) % size_of_waypoints]
        distance_between_waypoints = self._distance_between_waypoints[min_dist_idx]
        wp_delta_vector = after_waypoint.location - prev_waypoint.location
        location_delta_vector = point - prev_waypoint.location
        projected_distance = np.dot(wp_delta_vector, location_delta_vector) / distance_between_waypoints
        
        return RoarPyWaypointsProjection(
            min_dist_idx,
            projected_distance
        )

    def trace_forward_projection(self, projection : RoarPyWaypointsProjection, distance : float) -> RoarPyWaypointsProjection:
        """
        Trace forward from a projection result
        :param projection: projection result
        :param distance: distance to trace forward
        :return: new projection result
        """
        size_of_waypoints = len(self.waypoints)
        assert size_of_waypoints > 1
        
        distance %= self._total_distance

        if distance >= 0:
            current_projection = copy.copy(projection)
            remaining_distance = distance
            while remaining_distance > 0:
                to_progress = self._distance_between_waypoints[current_projection.waypoint_idx] - current_projection.distance_from_waypoint
                if remaining_distance < to_progress:
                    current_projection = RoarPyWaypointsProjection(
                        current_projection.waypoint_idx,
                        current_projection.distance_from_waypoint + remaining_distance
                    )
                    remaining_distance = 0
                    break
                else:
                    remaining_distance -= to_progress
                    current_projection = RoarPyWaypointsProjection(
                        (current_projection.waypoint_idx + 1) % size_of_waypoints,
                        0
                    )
                    continue
            return current_projection
        else:
            current_projection = copy.copy(projection)
            remaining_distance = -distance
            while remaining_distance > 0:
                to_progress = current_projection.distance_from_waypoint
                if remaining_distance < to_progress:
                    current_projection = RoarPyWaypointsProjection(
                        current_projection.waypoint_idx,
                        current_projection.distance_from_waypoint - remaining_distance
                    )
                    remaining_distance = 0
                    break
                else:
                    remaining_distance -= to_progress
                    current_projection = RoarPyWaypointsProjection(
                        (current_projection.waypoint_idx - 1) % size_of_waypoints,
                        self._distance_between_waypoints[(current_projection.waypoint_idx - 1) % size_of_waypoints]
                    )
                    continue
            if current_projection.distance_from_waypoint == self._distance_between_waypoints[current_projection.waypoint_idx]:
                current_projection.waypoint_idx = (current_projection.waypoint_idx + 1) % size_of_waypoints
                current_projection.distance_from_waypoint = 0
            
            return current_projection
    
    def delta_distance_projection(self, projection_origin : RoarPyWaypointsProjection, projection_destination : RoarPyWaypointsProjection) -> float:
        dist_origin = self.total_distance_from_first_waypoint(projection_origin)
        dist_destination = self.total_distance_from_first_waypoint(projection_destination)
        delta_dist = (dist_destination - dist_origin + self._total_distance) % self._total_distance
        if delta_dist > self._total_distance / 2:
            delta_dist -= self._total_distance
        return delta_dist

    def get_interpolated_waypoint(self, projection: RoarPyWaypointsProjection) -> RoarPyWaypoint:
        prev_wp = self.waypoints[projection.waypoint_idx]
        next_wp = self.waypoints[(projection.waypoint_idx + 1) % len(self.waypoints)]
        alpha = np.clip(projection.distance_from_waypoint / self._distance_between_waypoints[projection.waypoint_idx], 0, 1)
        return RoarPyWaypoint.interpolate(prev_wp, next_wp, alpha)
    
    def total_distance_from_first_waypoint(
        self,
        projection_result: RoarPyWaypointsProjection
    ) -> float:
        return self._total_distance_from_first_waypoint[projection_result.waypoint_idx] + projection_result.distance_from_waypoint