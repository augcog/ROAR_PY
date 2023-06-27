import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
from typing import List, Optional
from serde import serde
from dataclasses import dataclass
from waypoint import RoarPyWaypoint

@serde
@dataclass
class RoarPyOccupancyMap:
    def __init__(self, size: int, resolution: float = 1.0, waypoints: Optional[List[RoarPyWaypoint]] = None):
        self.size = size
        self.resolution = resolution
        self.map = np.zeros((int(size * resolution), int(size * resolution)))
        
        if waypoints is not None:
            self.add_waypoints(waypoints)

    def add_road(self, waypoint1: RoarPyWaypoint, waypoint2: RoarPyWaypoint):
        # Get the vertices of the polygon representing the road
        point1, point2 = waypoint1.line_representation
        point3, point4 = waypoint2.line_representation[::-1]
        vertices = np.vstack([point1, point2, point3, point4])
        
        # Convert the coordinates to pixel space
        vertices *= self.resolution
        vertices = vertices.astype(int)

        rr, cc = polygon(vertices[:, 0], vertices[:, 1])
        
        # Ensure the coordinates are within the bounds of the map
        rr, cc = rr.clip(0, int(self.size * self.resolution) - 1), cc.clip(0, int(self.size * self.resolution) - 1)

        # Mark these points on the map as road
        self.map[rr, cc] = 1

    def add_waypoints(self, waypoints: List[RoarPyWaypoint]):
        for waypoint1, waypoint2 in zip(waypoints[:-1], waypoints[1:]):
            self.add_road(waypoint1, waypoint2)

    def display(self):
        plt.imshow(self.map, cmap='Greys', origin='lower')
        plt.show()
