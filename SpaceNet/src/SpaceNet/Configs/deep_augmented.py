from msgspec import Struct


class Config(Struct):
    eps_roots: float = 1e-5
    eps_rcov: float = 1.0
    base_name: str = "classic"
    model_dir: str = "."
