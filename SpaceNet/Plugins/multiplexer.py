from SpaceNet.Plugins.plugin import Ports, Plugin


class Multiplexer(Plugin):


    def __init__(self):
        self.input_ports: Ports = {}
        self.output_ports: Ports = {}


    def execute(self) -> None:

        for o_id, output_port in self.output_ports.items():
            for i_id, input_port in self.input_ports.items():
                if o_id == i_id:
                    output_port.value = input_port.value