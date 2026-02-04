"""
Persist the configuration on the disk.
Useful f.e. for Deep Augmented Approaches where the engine is trained by a certain configuration.
"""
from typing import Protocol


class Configuration[T](Protocol):


    def save(self, path: str):...


    def load(self, path: str) -> T:...
