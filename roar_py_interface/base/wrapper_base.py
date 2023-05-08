from typing import Any, TypeVar, Generic, Union, Callable
import threading

from roar_py_interface.base.wrapper_base import RoarPyWrapper
from ..actors import RoarPyActor
from ..sensors import RoarPySensor
from ..worlds import RoarPyWorld

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
    def __init__(self, wrapped_object: Union[_TSWrapped,RoarPyWrapper[_TSWrapped]], tslock : threading.RLock):
        super().__init__(wrapped_object, wrapper_name="ThreadSafeWrapper")
        if not hasattr(self.unwrapped, "_tslock"):
            setattr(self.unwrapped, "_tslock", tslock)
        else:
            raise RuntimeError("Cannot wrap object twice")

def roar_py_thread_sync(func):
    def func_wrapper(self, *args, **kwargs):
        if hasattr(self, "_tslock"):
            with self._tslock:
                func_ret = func(self, *args, **kwargs)
                if isinstance(func_ret, RoarPyActor) or isinstance(func_ret, RoarPySensor) or isinstance(func_ret, RoarPyWorld):
                    func_ret = RoarPyThreadSafeWrapper(func_ret, self._tslock)
        else:
            func_ret = func(self, *args, **kwargs)
        return func_ret
    func_wrapper.is_thread_sync = True
    return func_wrapper

_ItemWrapped = TypeVar("_ItemWrapped")
class RoarPyAddItemWrapper(RoarPyWrapper[_ItemWrapped], Generic[_ItemWrapped]):
    def __init__(self, wrapped_object: Union[_ItemWrapped, RoarPyWrapper[_ItemWrapped]], add_callback : Callable[[Any], None] = None, remove_callback : Callable[[Any], None] = None):
        super().__init__(wrapped_object, wrapper_name="AddItemWrapper")
        if not hasattr(self.unwrapped, "_append_item_cb"):
            if add_callback is not None:
                setattr(self.unwrapped, "_append_item_cb", add_callback)
        else:
            raise RuntimeError("Cannot wrap object twice")
        
        if not hasattr(self.unwrapped, "_remove_item_cb"):
            if remove_callback is not None:
                setattr(self.unwrapped, "_remove_item_cb", remove_callback)
        else:
            raise RuntimeError("Cannot wrap object twice")

def roar_py_append_item(func):
    def func_wrapper(self, *args, **kwargs):
        func_ret = func(self, *args, **kwargs)
        if hasattr(self, "_append_item_cb"):
            self._append_item_cb(func_ret)
        return func_ret
    func_wrapper.is_append_item = True
    return func_wrapper

def roar_py_remove_item(func):
    def func_wrapper(self, arg1, *args, **kwargs):
        func_ret = func(self, arg1, *args, **kwargs)
        if hasattr(self, "_remove_item_cb"):
            self._remove_item_cb(arg1)
        return func_ret
    func_wrapper.is_remove_item = True
    return func_wrapper