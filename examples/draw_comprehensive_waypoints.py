import roar_py_carla
import roar_py_interface
import carla
import numpy as np
import asyncio
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import transforms3d as tr3d

async def main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(15.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    
    carla_world = roar_py_instance.world
    carla_world.set_asynchronous(True)
    carla_world.set_control_steps(0.00, 0.005)

    print("Map Name", carla_world.map_name)
    comprehensive_waypoints = roar_py_instance.world.comprehensive_waypoints
    spawn_points = roar_py_instance.world.spawn_points
    roar_py_instance.close()
    
    with plt.ion():
        for spawn_point in spawn_points:
            spawn_point_heading = tr3d.euler.euler2mat(0,0,spawn_point[1][2]) @ np.array([1,0,0])
            plt.arrow(
                spawn_point[0][0], 
                spawn_point[0][1], 
                spawn_point_heading[0] * 50, 
                spawn_point_heading[1] * 50, 
                width=10, 
                color='r'
            )
        for lane_id, waypoint_list in comprehensive_waypoints.items():
            for waypoint in waypoint_list:
                rep_line = waypoint.line_representation
                rep_line = np.asarray(rep_line)
                waypoint_heading = tr3d.euler.euler2mat(*waypoint.roll_pitch_yaw) @ np.array([1,0,0])
                plt.arrow(
                    waypoint.location[0], 
                    waypoint.location[1], 
                    waypoint_heading[0] * 0.5, 
                    waypoint_heading[1] * 0.5, 
                    width=0.5, 
                    color='r'
                )
                if lane_id == 1:
                    plt.plot(rep_line[:,0], rep_line[:,1], label="Lane {}".format(lane_id), color='r')
                else:
                    plt.plot(rep_line[:,0], rep_line[:,1], label="Lane {}".format(lane_id), color='b')
        
    plt.show()

if __name__ == '__main__':
    asyncio.run(main())