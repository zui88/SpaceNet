from SpaceNet.Utils.DeepAugmented.LossFunctions.rmspe_loss import RMSPELoss
from SpaceNet.Utils.DeepAugmented.LossFunctions.loss import PermutatedLoss

from tensorflow import keras

from typing import Any, Protocol, Callable


type ModelName = str
type Dataset = Any
type Percent = float
type TransformerClass = type[Any]


class DeepAugment(Protocol):


    def register_transformer(self, transformer: Callable[[Dataset], Dataset] | TransformerClass, **args):
        """

        Parameters
        ----------
        transformer
            either a callable function or a constructable type

        **args
            optional arguments for construction the transformer

        Returns
        -------

        """
        pass


    def fit_transformer(self, dataset: Dataset) -> None:...


    @property
    def transformer(self) -> Callable[[Dataset], Dataset] | None:...


    def register_models(self, model_descriptions: dict[ModelName, keras.models.Model]) -> None:...


    def register_model(self, name: ModelName, model: keras.models.Model) -> None:...


    def get_models(self) -> dict[ModelName, keras.models.Model] | None:...


    def load_models(self):...


    def save_models(self):...


    def conditionally_save(self, threshold: Percent = 0.) -> bool:...


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
    ) -> keras.callbacks.History:...
