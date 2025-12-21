import datetime
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class MessageType(Enum):
    """Tipi di messaggi supportati"""
    DATA = "data"
    COMMAND = "command"
    CONFIG = "config"
    STATUS = "status"
    ERROR = "error"
    FLOW_DETECTED = "flow_detected"

    @classmethod
    def from_str(cls, value: str) -> "MessageType":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"MessageType non valido: {value!r}. "
                f"Valori ammessi: {[e.value for e in cls]}"
            )


@dataclass(slots=True)
class GUIMessage:
    type: MessageType
    data: Any = None

    def to_dict(self, **kwargs) -> dict:
        return {
            "type": self.type.value,
            "data": serialize_data(self.data, **kwargs),
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(**kwargs), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "GUIMessage":
        return cls(
            type=MessageType(d["type"]),
            data=deserialize_data(d.get("data")),
        )


def deserialize_data(obj: Any):
    if not isinstance(obj, dict):
        return obj

    t = obj.get("__type__")

    if t == "DataFrame":
        return pd.DataFrame(obj["value"])

    if t == "Series":
        return pd.Series(obj["value"])

    if t == "datetime":
        return datetime.fromisoformat(obj["value"])

    return obj


def serialize_data(obj: Any, **kwargs) -> Any:
    if obj is None:
        return None

    orient = kwargs.get("orient", "records")
    date_format = kwargs.get("date_format", "iso")
    index = kwargs.get("index", False)

    # pandas
    if isinstance(obj, pd.DataFrame):

        return {
            "__type__": "DataFrame",
            "value": obj.to_json(orient=orient, date_format=date_format, index=index),
        }

    if isinstance(obj, pd.Series):
        return {
            "__type__": "Series",
            "value": obj.to_json(orient="records", date_format="iso"),
        }

    # datetime
    if isinstance(obj, datetime):
        return {
            "__type__": "datetime",
            "value": obj.isoformat(),
        }

    # numpy
    if isinstance(obj, np.generic):
        return obj.item()

    # basic JSON types
    if isinstance(obj, (str, int, float, bool, list, dict)):
        return obj

    raise TypeError(f"Non serializzabile: {type(obj)}")
