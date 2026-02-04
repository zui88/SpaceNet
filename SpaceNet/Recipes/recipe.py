from SpaceNet.Plugins.plugin import Ports
from SpaceNet.Utils.graph import Graph
from SpaceNet.Utils.id import ID
from typing import Any


class Recipe(Graph):


    input_ports: Ports
    output_ports: Ports


    def __init__(self):
        super().__init__()


    def run(self, **inputs) -> dict[ID, Any]:
        """
        A user of 'Recipe' should use the 'run' method.  This method takes care of the input data mapping.
        All underlying plugins or embedded recipes will be executed by a defined order specified by this recipe.

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        return super().evaluate(**inputs)


    @property
    def inputs(self) -> dict[ID, Any]:
        return self.nodes["input"].input_ports


    @property
    def outputs(self) -> dict[ID, Any]:
        return self.nodes["output"].output_ports


    def execute(self) -> None:
        """
        This function will be mostly executed from another upper recipe where the actual recipe is the 'child'.
        Here the input and output ports of a plugin will be set up for correct data flow.

        Returns
        -------

        """
        inputs = {id: link.value for id, link in self.inputs.items()}
        outputs = self.run(**inputs)
        for id, value in outputs.items():
            self.outputs[id].value = value
