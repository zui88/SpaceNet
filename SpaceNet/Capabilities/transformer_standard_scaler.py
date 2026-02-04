from typing import Any


type Dataset = Any


class StandardScaler:


    def fit(self, x: Dataset) -> None:...


    def fit_transform(self, x: Dataset) -> Dataset:...


    def transform(self, x: Dataset) -> Dataset:...


    def is_fitted(self) -> bool:...


    def save(self, path: str):...


    @classmethod
    def load(cls, path: str):...
