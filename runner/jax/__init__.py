__all__ = ["JAXTrainer"]


def __getattr__(name):
    if name == "JAXTrainer":
        from .trainer import JAXTrainer

        return JAXTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
