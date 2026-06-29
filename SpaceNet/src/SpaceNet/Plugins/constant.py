from SpaceNet.Plugins.plugin import Ports, Plugin, Link


class Constant(Plugin):
    def __init__(self, value):
        self.input_ports: Ports = {}
        self.output_ports: Ports = {"const": Link()}

        self.value = value

    def execute(self) -> None:

        self.output_ports["const"].value = self.value
