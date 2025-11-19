"""
Ingredient NER package: training, normalization & encoding utilities.
"""

# Public config interface: three global namespaces + helpers
from .config import (
    DATA,
    TRAIN,
    OUT,
    load_configs_from_yaml,
    load_configs_from_dict,
    print_configs,
)

__all__ = [
    "DATA",
    "TRAIN",
    "OUT",
    "load_configs_from_yaml",
    "load_configs_from_dict",
    "print_configs",
]
