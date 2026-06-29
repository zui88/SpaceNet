from SpaceNet.Synthesizer.signal import (
    SignalGenerator,
    ChirpSignal,
    SincTSignal,
    RandomSignal,
)
from SpaceNet.Synthesizer.geometry import (
    ArrayGeometry,
    ULAArray,
    SteeringType,
    RandomArray,
)
from msgspec import Struct, field
from enum import Enum


class SignalKind(Enum):
    CHIRP_SIGNAL = 1
    SINCT_SIGNAL = 2
    RANDOM_SIGNAL = 3


class ArrayKind(Enum):
    RANDOM_ARRAY = 1
    ULA_ARRAY = 2


class Array(Struct):
    antennas: int = 6
    d_lambda: float = 1.0
    kind: ArrayKind = ArrayKind.RANDOM_ARRAY


class Signal(Struct):
    alpha: float = 1.0
    T: float = 0.5
    fs: int = 25
    n_samples: int | None = None
    kind: SignalKind = SignalKind.CHIRP_SIGNAL


class Config(Struct):
    scan_range: int = 360
    d_sources: int | None = None
    inference_mode: bool = True
    snr_db: float | tuple[float, float] = 30.0
    array: Array = field(default_factory=Array)
    signal: Signal = field(default_factory=Signal)

    @property
    def signal_provider(self) -> SignalGenerator:
        inputs = {
            "n_samples": self.signal.n_samples,
            "T": self.signal.T,
            "fs": self.signal.fs,
        }

        match self.signal.kind:
            case SignalKind.SINCT_SIGNAL:
                return SincTSignal(**inputs)
            case SignalKind.RANDOM_SIGNAL:
                return RandomSignal(**inputs)
            case SignalKind.CHIRP_SIGNAL:
                return ChirpSignal(self.signal.alpha, **inputs)
            case _:
                return RandomSignal(**inputs)

    @property
    def array_geometry(self) -> ArrayGeometry:
        match self.array.kind:
            case ArrayKind.RANDOM_ARRAY:
                return RandomArray(antennas=self.array.antennas)
            case ArrayKind.ULA_ARRAY:
                return ULAArray(
                    antennas=self.array.antennas, d_lambda=self.array.d_lambda
                )
            case _:
                return RandomArray(antennas=self.array.antennas)

    @property
    def steering(self) -> SteeringType:
        return ULAArray(antennas=self.array.antennas).get_steering
