from SpaceNet.Capabilities.deep_augment import ModelName, Dataset, Percent, TransformerClass
from SpaceNet.Utils.DeepAugmented.LossFunctions.loss import PermutatedLoss
from SpaceNet.Utils.DeepAugmented.LossFunctions.rmspe_loss import RMSPELoss
from SpaceNet.Engines.engine import Engine

from typing import Callable
from pathlib import Path
from typing import Any
import inspect

from tensorflow import keras
import tensorflow as tf


class EngineType(keras.Model, Engine):...


class DeepAugmentV1:


    def __init__(self, engine: EngineType, network_names: list[str], base_name: str, base_path: str, version: str = "modelv1"):
        self.engine  = engine
        self.workdir = base_path
        self.paths   = [base_path + "/" + version + "_" + base_name + "_" + name + ".keras" for name in network_names]
        self.name_paths: dict[ModelName, Path] = {name: Path(path) for name, path in zip(network_names, self.paths)}
        self.models: dict[ModelName, keras.models.Model] | None = None
        self._transformer: Callable[[Dataset], Dataset] | TransformerClass | None = lambda x: x
        self.transformer_args: dict[str, Any] = {}


    def register_transformer(self, transformer: Callable[[Dataset], Dataset] | TransformerClass, **args):
        self._transformer     = transformer
        self.transformer_args = args


    def fit_transformer(self, dataset: Dataset) -> None:
        if inspect.isclass(self._transformer):
            self._transformer = self._transformer(**self.transformer_args)

        if hasattr(self._transformer, "fit"):
            self._transformer.fit(dataset)


    @property
    def transformer(self) -> Callable[[Dataset], Dataset] | None:
        if inspect.isclass(self._transformer):
            self._transformer = self._transformer(**self.transformer_args)

        if hasattr(self._transformer, "transform"):
            if hasattr(self._transformer, 'is_initialized') and not self._transformer.is_initialized():
                if hasattr(self._transformer, 'load'):
                    self._transformer = self._transformer.load(self.workdir + "/" + "scaler.npz")
            return self._transformer.transform

        return self._transformer


    def get_models(self) -> dict[ModelName, keras.models.Model] | None:
        return self.models


    def load_models(self):
        self.models = {name: keras.models.load_model(path) for name, path in self.name_paths.items()}


    def save_models(self):
        if self.models is not None:
            for path, model in zip(self.name_paths.values(), self.models.values()):
                keras.models.save_model(model, path)
            if hasattr(self._transformer, "save"): self._transformer.save(self.workdir + "/" + "scaler.npz")
        else:
            raise ValueError(f"Models unloaded")


    def train_models(self,
                     input_data: Dataset,
                     output_data: Dataset,
                     validation_data: tuple[Dataset, Dataset] | None = None,
                     freezing_layers: tuple[bool, ...] | None = None,
                     debuggable: bool = False,
                     epochs: int = 200,
                     batch_size: int = 32,
                     learning_rate: float = 0.0001,
                     patience: int | None = None,
                     factor: float = 0.25,
                     loss: PermutatedLoss = RMSPELoss()
                     ) -> keras.callbacks.History:
        """

        Parameters
        ----------
        factor
        patience
        learning_rate
        input_data
        output_data
        validation_data
        freezing_layers
        debuggable
        epochs
        batch_size

        loss : PermutatedLoss = RMSPELoss()
            The default loss is suitable for doa estimation.

        Returns
        -------

        """

        if self.models is None:
            raise ValueError("Models unloaded")

        for name, model in self.models.items():
            setattr(self.engine, name, model)

        if freezing_layers is not None:
            for model, frozen in zip(self.models.values(), freezing_layers):
                if frozen:
                    model.trainable = False

        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=1e-9,
        )

        if debuggable: tf.config.run_functions_eagerly(True)
        self.engine.compile(
            optimizer=optimizer,
            loss=loss,
            run_eagerly=debuggable,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience*3,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=patience,
                factor=factor,
                verbose=1,
            ),
        ] if patience is not None else None

        self.fit_transformer(input_data)
        if validation_data is not None:
            validation_data = (self.transformer(validation_data[0]), validation_data[1])

        history = self.engine.fit(
            self.transformer(input_data) if self.transformer is not None else input_data,
            output_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        setattr(self, "history", history)
        return history


    def register_models(self, model_descriptions: dict[ModelName, keras.models.Model]) -> None:
        """
        Overrides the internal registry.

        Parameters
        ----------
        model_descriptions

        Returns
        -------

        """
        self.models = model_descriptions


    def conditionally_save(self, threshold: Percent = 0.) -> bool:
        if hasattr(self, "history"):
            history          = self.history
            initial_val_loss = history.history["val_loss"][0]
            final_val_loss   = history.history["val_loss"][-1]
            improvement      = (initial_val_loss - final_val_loss) / initial_val_loss * 100.0

            if improvement > threshold:
                self.save_models()
                return True

        return False


    def register_model(self, name: ModelName, model: keras.models.Model) -> None:
        """
        Extents or create a new registry.

        Parameters
        ----------
        name
        model

        Returns
        -------

        """
        if self.models is not None:
            self.models[name] = model
        else:
            self.models = {name: model}