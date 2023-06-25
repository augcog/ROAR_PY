import numpy as np
from matplotlib import pyplot as plt

class RoarPyOccupancyMap:
    map: np.ndarray
    size: int

    def __init__(self, size) -> None:
        self.size = size
        self.map = np.zeros((size, size))

    def add_waypoint(self, waypoint) -> None:
        global_coordinate_pos, global_coordinate_neg = waypoint.line_representation
        global_coordinate_pos = global_coordinate_pos[:2].astype(int)
        global_coordinate_neg = global_coordinate_neg[:2].astype(int)
        if (0 <= global_coordinate_pos[0] < self.size) and (0 <= global_coordinate_pos[1] < self.size):
            self.map[global_coordinate_pos[0], global_coordinate_pos[1]] = 1
        else:
            print("Waypoint out of bounds")
        if (0 <= global_coordinate_neg[0] < self.size) and (0 <= global_coordinate_neg[1] < self.size):
            self.map[global_coordinate_neg[0], global_coordinate_neg[1]] = 1
        else:
            print("Waypoint out of bounds")


    def add_waypoints(self, waypoints) -> None:
        for waypoint in waypoints:
            self.add_waypoint(waypoint)

    def display(self) -> None:
        plt.imshow(self.map)
        plt.show()

    def save(self, path) -> None:
        np.save(path, self.map)
