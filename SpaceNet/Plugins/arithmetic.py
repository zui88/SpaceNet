from operator import add, mul, sub
from typing import Callable, Any

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class BinaryOperation(Plugin):


    def __init__(self, operation: Callable[[Any, Any], Any]):
        self.operation = operation
        self.input_ports: Ports = {
            "a": Link(),
            "b": Link(),
        }
        self.output_ports: Ports = {
            "result": Link(),
        }


    def execute(self) -> None:
        self.output_ports["result"].value = self.operation(
            self.input_ports["a"].value,
            self.input_ports["b"].value,
        )


class Addition(BinaryOperation):


    def __init__(self):
        super().__init__(add)


class Substract(BinaryOperation):


    def __init__(self):
        super().__init__(sub)


class Multiply(BinaryOperation):


    def __init__(self):
        super().__init__(mul)
