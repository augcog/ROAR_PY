from typing import Any, TypeVar, Generic, Union
import threading

_Wrapped = TypeVar("_Wrapped")
class RoarPyWrapper(Generic[_Wrapped]):
    def __init__(self, wrapped_object: Union[_Wrapped,"RoarPyWrapper[_Wrapped]"], wrapper_name: str):
        self._wrapped_object = wrapped_object
        self._wrapper_name = wrapper_name
    
    def __getattr__(self, __name: str) -> Any:
        if __name.startswith("_"):
            raise AttributeError("Cannot access private attribute")
        return getattr(self._wrapped_object, __name)
    
    def __str__(self) -> str:
        return self._wrapper_name + "(" + self._wrapped_object.__str__() + ")"

    @property
    def unwrapped(self) -> _Wrapped:
        if isinstance(self._wrapped_object, RoarPyWrapper):
            return self._wrapped_object.unwrapped
        else:
            return self._wrapped_object

_TSWrapped = TypeVar("_TSWrapped")
class RoarPyThreadSafeWrapper(RoarPyWrapper[_TSWrapped], Generic[_TSWrapped]):
    def __init__(self, wrapped_object: Union[_TSWrapped,"RoarPyThreadSafeWrapper[_TSWrapped]"], tslock : threading.RLock):
        super().__init__(wrapped_object, wrapper_name="ThreadSafeWrapper")
        setattr(self.unwrapped, "_tslock", tslock)
    
    def __getattr__(self, __name: str) -> Any:
        if __name.startswith("_"):
            raise AttributeError("Cannot access private attribute")
        return getattr(self._wrapped_object, __name)
    
    def __str__(self) -> str:
        return self._wrapper_name + "(" + self._wrapped_object.__str__() + ")"

    @property
    def unwrapped(self) -> _TSWrapped:
        if isinstance(self._wrapped_object, RoarPyThreadSafeWrapper):
            return self._wrapped_object.unwrapped
        else:
            return self._wrapped_object

def roar_py_thread_sync(func):
    def func_wrapper(self, *args, **kwargs):
        if hasattr(self, "_tslock"):
            with self._tslock:
                return func(self, *args, **kwargs)
    return func_wrapper