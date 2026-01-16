# core/quantization/strategies/__init__.py

_STRATEGY_REGISTRY = {}

def register_strategy(name):
    """
    Decorator to register a calibration strategy class.

    Args:
        name (str): The unique identifier for the strategy (e.g., 'streaming_audio').
    """
    def decorator(cls):
        _STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator

def get_strategy_class(name):
    """
    Retrieve a strategy class by its name.
    """
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"Unknown calibration strategy: '{name}'. "
                         f"Available strategies: {list(_STRATEGY_REGISTRY.keys())}")
    return _STRATEGY_REGISTRY[name]