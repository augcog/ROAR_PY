import carla
from typing import Dict, List, Optional, Union
from roar_py_interface.wrappers import roar_py_thread_sync
from ..worlds import RoarPyCarlaWorld
import weakref

class RoarPyCarlaInstance:
    actor_to_instance_map : Dict[int,"RoarPyCarlaBase"] = {}

    def __init__(
        self,
        carla_client: carla.Client,
        world_override : Optional[RoarPyCarlaWorld] = None
    ):
        self.carla_client = carla_client
        if world_override is not None:
            self.world = world_override
        else:
            self.world = RoarPyCarlaWorld(carla_client.get_world(),self)

    @roar_py_thread_sync
    def __refresh_world(self):
        self.world = RoarPyCarlaWorld(self.carla_client.get_world(),self)

    """
    Creates a new world with using map_name map. All actors in the current world will be destroyed. 
    """
    @roar_py_thread_sync
    def load_world(self, map_name : str, reset_settings : bool=True):
        self.__cleanup_actor_instance_map()
        self.carla_client.load_world(map_name, reset_settings)
        self.__refresh_world()
    
    """
    Reloads the current world. All actors in the current world will be destroyed.
    """
    @roar_py_thread_sync
    def reload_world(self, reset_settings : bool=True):
        self.__cleanup_actor_instance_map()
        self.carla_client.reload_world(reset_settings)
        self.__refresh_world()
    
    @roar_py_thread_sync
    def get_available_maps(self) -> List[str]:
        return self.carla_client.get_available_maps()
    
    def get_client_version(self) -> str:
        return self.carla_client.get_client_version()
    
    @roar_py_thread_sync
    def get_server_version(self) -> str:
        return self.carla_client.get_server_version()

    @roar_py_thread_sync
    def __cleanup_actor_instance_map(self):
        print("ROAR_PY_CARLA: Cleaning up actor instance map")
        all_actors = list(self.actor_to_instance_map.values())
        for actor in all_actors:
            try:
                if not actor.is_closed():
                    actor.close()
            except Exception:
                # in case actor is already closed
                pass
        
        self.actor_to_instance_map.clear()
    
    @roar_py_thread_sync
    def register_actor(self, actor_id : int, actor_instance : "RoarPyCarlaBase"):
        self.actor_to_instance_map[actor_id] = weakref.proxy(actor_instance, lambda x: self.unregister_actor(actor_id, x))
    
    @roar_py_thread_sync
    def unregister_actor(self, actor_id : int, actor_instance : "RoarPyCarlaBase"):
        if self.actor_to_instance_map.get(actor_id, None) is actor_instance:
            del self.actor_to_instance_map[actor_id]
    
    def search_actor(self, actor_id : int) -> Optional["RoarPyCarlaBase"]:
        return self.actor_to_instance_map.get(actor_id, None)
    
    def close(self):
        self.__cleanup_actor_instance_map()
    
    def is_closed(self) -> bool:
        return len(self.actor_to_instance_map) == 0
    
    def clean_actors_not_registered(self, typeid_wildcard : Optional[Union[str, List[str]]] = ["vehicle.*", "sensor.*"]):
        native_actors : carla.ActorList = self.world.carla_world.get_actors()
        
        if typeid_wildcard is not None:
            if isinstance(typeid_wildcard, str):
                to_iterate = [native_actors.filter(typeid_wildcard)]
            else:
                to_iterate = [native_actors.filter(x) for x in typeid_wildcard]
        else:
            to_iterate = [native_actors]
        
        instance_map_keys = self.actor_to_instance_map.keys()
        for c_native_actors in to_iterate:
            for actor in c_native_actors:
                if actor.id in instance_map_keys:
                    continue
                else:
                    print("ROAR_PY_CARLA: Cleaning up actor: {}, {}".format(actor.id, actor.type_id))
                    if hasattr(actor, "is_listening") and hasattr(actor, "stop") and actor.is_listening:
                        actor.stop()
                    actor.destroy()

    def __del__(self):
        self.close()