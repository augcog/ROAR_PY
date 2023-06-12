import websockets
from roar_py_interface.base import RoarPyRemoteSupportedSensorData, RoarPySensor, RoarPyRemoteSupportedSensorSerializationScheme
from serde import serde
from dataclasses import dataclass
import enum

class RoarPyWebsocketCommunicationCommand(enum.IntEnum):
    CLIENT_TRY_READ_VEHICLE = 0
    CLIENT_READ_SENSOR_OBSERVATION = 1

@serde
@dataclass
class RoarPyWebsocketCommunicationData:
    command: RoarPyWebsocketCommunicationCommand


