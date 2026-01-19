# Copyright (c) 2026 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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