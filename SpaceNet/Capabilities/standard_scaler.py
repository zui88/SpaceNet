from typing import Any

import numpy as np


type Dataset = Any


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        scaled = (X - self.mean) / self.std
        return scaled

    def save(self, path_name):
        np.savez(path_name, mean=self.mean, std=self.std)

    def is_initialized(self) -> bool:
        return self.mean is not None

    @classmethod
    def load(cls, path_name):
        data_dict = np.load(path_name)
        scaler = cls()
        scaler.mean = data_dict["mean"]
        scaler.std = data_dict["std"]

        return scaler
