class DummyScaler:
    @staticmethod
    def fit(X):
        pass

    @staticmethod
    def transform(X):
        return X

    @staticmethod
    def is_initialized() -> bool:
        return True
