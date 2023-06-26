import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
from typing import List
from waypoint import RoarPyWaypoint  # Assuming RoarPyWaypoint class is in RoarPyWaypoint.py

class OccupancyMap:
    def __init__(self, size):
        self.size = size
        self.map = np.zeros((size, size), dtype=np.uint8)

    def add_road(self, waypoint1: RoarPyWaypoint, waypoint2: RoarPyWaypoint):
        # Get the vertices of the polygon representing the road
        vertices = np.vstack([waypoint1.line_representation, waypoint2.line_representation[::-1]])
        rr, cc = polygon(vertices[:,0], vertices[:,1])
        
        # Ensure the coordinates are within the bounds of the map
        rr, cc = rr.clip(0, self.size-1), cc.clip(0, self.size-1)

        # Mark these points on the map as road
        self.map[rr, cc] = 1

    def add_waypoints(self, waypoints: List[RoarPyWaypoint]):
        for waypoint1, waypoint2 in zip(waypoints[:-1], waypoints[1:]):
            self.add_road(waypoint1, waypoint2)

    def display(self):
        plt.imshow(self.map, cmap='Greys')
        plt.show()
