from SpaceNet.Capabilities.deep_augment import DeepAugment
from SpaceNet.Engines.engine import RetDD, RetDoa

from tensorflow import keras


def make_trainable[T](cls: type[T]) -> type[T]:
    """
    Extend the given class to be a keras model that supports training.  The 'trainable' class
    takes care of passing the arguments to the right subclass and also initializes the keras
    model.

    Parameters
    ----------
    cls
        concrete engine type

    Returns
    -------
    DoaEngine
        The returned engine is 'silently' trainable, though the user can also implicitly issue
        keras methods/functions even if those are hidden.

    """
    class TrainableDoaEngine(keras.Model, cls):


        def __init__(self, *args, **inputs) -> None:
            keras.Model.__init__(self)
            cls.__init__(self, *args, **inputs)
            self.transformer = None


        def call(self, inputs):
            r_sensed    = inputs
            predictions = self.estimate(r_sensed=r_sensed)
            return predictions[0].raw


        def estimate(self, **inputs) -> RetDoa | RetDD:
            return cls.estimate(self, **self.transform(**inputs))


        def transform(self, r_sensed):
            if self.transformer is None:
                self.transformer = self.capability_registry[DeepAugment].transformer
                if self.transformer is None: raise NotImplementedError("No transformer registered")

            scaled = self.transformer(r_sensed)
            return {"r_sensed":scaled}


    return TrainableDoaEngine
