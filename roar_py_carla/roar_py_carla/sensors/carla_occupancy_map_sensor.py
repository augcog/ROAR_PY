from roar_py_interface.sensors.occupancy_map_sensor import RoarPyOccupancyMapSensor, RoarPyOccupancyMapSensorData
from roar_py_interface.worlds.occupancy_map import RoarPyOccupancyMapProducer
import typing
import gymnasium as gym

class RoarPyCarlaOccupancyMapSensor(RoarPyOccupancyMapSensor):
    def __init__(self, producer : RoarPyOccupancyMapProducer, actor : "RoarPyCarlaActor", name: str = "carla_occupancy_map_sensor"):
        super().__init__(name)
        self._closed = False
        self.producer = producer
        self.actor = actor
        self._last_data : typing.Optional[RoarPyOccupancyMapSensorData] = None
    
    def get_gym_observation_spec(self) -> gym.Space:
        return RoarPyOccupancyMapSensorData.gym_observation_space(self.producer.width, self.producer.height)
    
    async def receive_observation(self) -> RoarPyOccupancyMapSensorData:
        location = self.actor.get_3d_location()[:2]
        yaw = self.actor.get_roll_pitch_yaw()[2]
        occupancy_map = self.producer.plot_occupancy_map(location, yaw)
        self._last_data = RoarPyOccupancyMapSensorData.from_image(occupancy_map)
        return self._last_data
    
    def get_last_observation(self) -> typing.Optional[RoarPyOccupancyMapSensorData]:
        return self._last_data

    def convert_obs_to_gym_obs(self, obs: RoarPyOccupancyMapSensorData):
        return obs.convert_obs_to_gym_obs()
    
    def close(self):
        self._closed = True
    
    def is_closed(self) -> bool:
        return self._closed
