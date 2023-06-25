import numpy as np
from matplotlib import pyplot as plt

class RoarPyOccupancyMap:
    map: np.ndarray
    size: int

    def __init__(self, size) -> None:
        self.size = size
        self.map = np.zeros((size, size))

    def add_waypoint(self, waypoint):
        global_coordinate_pos, global_coordinate_neg = waypoint.line_representation
        global_coordinate_pos = global_coordinate_pos[:2].astype(int)
        global_coordinate_neg = global_coordinate_neg[:2].astype(int)
        if global_coordinate_pos[0] < 0 \
            or global_coordinate_pos[0] >= self.size \
            or global_coordinate_pos[1] < 0 \
            or global_coordinate_pos[1] >= self.size:
            print("Waypoint out of bounds")
        else:
            self.map[global_coordinate_pos[0], global_coordinate_pos[1]] = 1
        if global_coordinate_neg[0] < 0 \
            or global_coordinate_neg[0] >= self.size \
            or global_coordinate_neg[1] < 0 \
            or global_coordinate_neg[1] >= self.size:
            print("Waypoint out of bounds")
        else:
            self.map[global_coordinate_neg[0], global_coordinate_neg[1]] = 1


    def add_waypoints(self, waypoints):
        for waypoint in waypoints:
            self.add_waypoint(waypoint)

    def display(self):
        plt.imshow(self.map)
        plt.show()

    def save(self, path):
        np.save(path, self.map)
