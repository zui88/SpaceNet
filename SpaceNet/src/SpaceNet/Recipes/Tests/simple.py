from SpaceNet.Utils.id import ID
from typing import Any
from SpaceNet.Plugins.arithmetic import Multiply, Substract, Addition
from SpaceNet.Recipes.recipe import Recipe


class Simple(Recipe):
    def __init__(self):
        super().__init__()

        #############################################
        # register the plugins
        #############################################
        self.register_node("mul", Multiply())
        self.register_node("sub", Substract())
        self.register_node("add", Addition())

        #############################################
        # connect the plugins
        #############################################
        self.connect_node("input", "a", "add", "a")
        self.connect_node("input", "b", "add", "b")
        self.connect_node("input", "a", "mul", "a")
        self.connect_node("input", "d", "sub", "b")

        self.connect_node("add", "result", "mul", "b")

        self.connect_node("mul", "result", "sub", "a")
        self.connect_node("mul", "result", "output", "e")

        self.connect_node("sub", "result", "output", "f")

    def run(self, **inputs) -> dict[ID, Any]:
        res = super().run(a=5, b=42, d=-6)
        print(f"test restult: {res}")
        return res


if __name__ == "__main__":
    Simple().run()
