import carla
from typing import Dict, List, Optional
from ..base import RoarPyCarlaBase
from ..worlds import RoarPyCarlaWorld

class RoarPyCarlaInstance:
    actor_to_instance_map : Dict[int,RoarPyCarlaBase] = {}

    def __init__(
        self,
        carla_client: carla.Client,
        world_override : Optional[RoarPyCarlaWorld] = None
    ):
        self.carla_client = carla_client
        if world_override is not None:
            self.world = world_override
        else:
            self.world = RoarPyCarlaWorld()

    def __refresh_world(self):
        self.world = RoarPyCarlaWorld()

    """
    Creates a new world with using map_name map. All actors in the current world will be destroyed. 
    """
    def load_world(self, map_name : str, reset_settings : bool=True):
        self.__cleanup_actor_instance_map()
        self.carla_client.load_world(map_name, reset_settings)
        self.__refresh_world()
    
    """
    Reloads the current world. All actors in the current world will be destroyed.
    """
    def reload_world(self, reset_settings : bool=True):
        self.__cleanup_actor_instance_map()
        self.carla_client.reload_world(reset_settings)
        self.__refresh_world()
    
    def get_available_maps(self) -> List[str]:
        return self.carla_client.get_available_maps()
    
    def get_client_version(self) -> str:
        return self.carla_client.get_client_version()
    
    def get_server_version(self) -> str:
        return self.carla_client.get_server_version()

    def __cleanup_actor_instance_map(self):
        for actor in self.actor_to_instance_map.values():
            if not actor.is_closed():
                actor.close()
        self.actor_to_instance_map = {}
    
    def register_actor(self, actor_id : int, actor_instance : RoarPyCarlaBase):
        self.actor_to_instance_map[actor_id] = actor_instance
    
    def unregister_actor(self, actor_id : int, actor_instance : RoarPyCarlaBase):
        if self.actor_to_instance_map.get(actor_id, None) == actor_instance:
            del self.actor_to_instance_map[actor_id]
    
    def search_actor(self, actor_id : int) -> Optional[RoarPyCarlaBase]:
        return self.actor_to_instance_map.get(actor_id, None)