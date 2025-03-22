import abc
from dataclasses import dataclass

import draccus
@dataclass
class MotorsBusConfig :
    serial_port: str
    ip_address: str
    motors: dict[str, tuple[int, str]]
    
