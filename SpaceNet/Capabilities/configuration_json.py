"""
Persist the configuration on the disk in JSON format.
Useful f.e. for Deep Augmented Approaches where the engine is trained by a certain configuration.
"""

from typing import overload

import msgspec


class JsonEncoderDecoder[T]:
    @overload
    def __init__(self, config: T, verbose: bool = False): ...

    @overload
    def __init__(self, config_type: type[T], verbose: bool = False): ...

    def __init__(self, args, verbose: bool = False) -> None:
        self.verbose = verbose

        if isinstance(args, type):
            self._config_type = args
            self.config: T | None = None
        else:
            self._config_type = type(args)
            self.config = args

    def save(self, path: str):
        if self.config is None:
            raise ValueError("configuration not set")

        encoded = msgspec.json.encode(self.config)
        if self.verbose:
            print(encoded)

        with open(path, "wb") as file:
            file.write(encoded)

    def load(self, path: str) -> T | None:
        with open(path, "rb") as file:
            encoded = file.read()

        self.config = msgspec.json.decode(encoded, type=self._config_type)
        if self.verbose:
            print(self.config)
        return self.config
