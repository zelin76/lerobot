import abc
from dataclasses import dataclass

import draccus
@dataclass
class MotorsBusConfig :
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False
