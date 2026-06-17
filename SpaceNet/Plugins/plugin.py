from SpaceNet.Utils.id import ID
from dataclasses import dataclass
from typing import Protocol, Any, TypeAlias


@dataclass
class Link:
    value: Any = None


Ports: TypeAlias = dict[ID, Link]


class Plugin(Protocol):


    input_ports: Ports
    output_ports: Ports


    @property
    def inputs(self) -> Ports:
        return self.input_ports


    @property
    def outputs(self) -> Ports:
        return self.output_ports


    def execute(self) -> None:
        ...
