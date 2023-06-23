from typing import TypeVar, Generic

_InMsgT = TypeVar("_InMsgT")
_OutMsgT = TypeVar("_OutMsgT")
class RoarPyObjectWithRemoteMessage(Generic[_InMsgT, _OutMsgT]):
    def _depack_info(self, data: _InMsgT) -> bool:
        raise NotImplementedError()
    
    async def _tick_remote(self) -> None:
        return

    def _pack_info(self) -> _OutMsgT:
        raise NotImplementedError()

def register_object_with_remote_message(
    in_msg_type: type,
    out_msg_type: type,
):
    def _register_object_with_remote_message(cls):
        cls._in_msg_type = in_msg_type
        cls._out_msg_type = out_msg_type
        return cls
    
    return _register_object_with_remote_message