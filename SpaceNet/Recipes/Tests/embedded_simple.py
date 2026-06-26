from SpaceNet.Recipes.Tests.simple import Simple
from SpaceNet.Recipes.recipe import Recipe


class EmbeddedSimple(Recipe):
    def __init__(self):
        super().__init__()

        self.register_node("test", Simple())

        self.connect_node("input", "ea", "test", "a")
        self.connect_node("input", "eb", "test", "b")
        self.connect_node("input", "ed", "test", "d")

        self.connect_node("test", "e", "output", "ee")
        self.connect_node("test", "f", "output", "ef")


if __name__ == "__main__":
    res = EmbeddedSimple().run(ea=5, eb=42, ed=-6)
    print(f"embedded result: {res}")
