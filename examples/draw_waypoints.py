import roar_py_carla_implementation
import roar_py_interface
import carla
import numpy as np
import asyncio
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

async def main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla_implementation.RoarPyCarlaInstance(carla_client)
    
    carla_world = roar_py_instance.world
    carla_world.set_control_steps(0.05, 0.005)
    carla_world.set_asynchronous(False)
    
    waypoints = roar_py_instance.world.maneuverable_waypoints
    for waypoint in waypoints:
        plt.plot(waypoint.location[0], waypoint.location[1], 'ro')
    plt.show()

if __name__ == '__main__':
    asyncio.run(main())