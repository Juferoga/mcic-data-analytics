"""Transformers Module - Transformadores texto a número."""

from .base import BaseEncoder
from .encoders import LabelEncoder, OneHotEncoder, TargetEncoder, HashEncoder

__all__ = [
    "BaseEncoder",
    "LabelEncoder",
    "OneHotEncoder",
    "TargetEncoder",
    "HashEncoder",
]
