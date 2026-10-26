"""
Decay, instrument, and noise simulation functions.

This module provides various functions for simulating exponential decays,
blobs, instrument response functions (IRF), gradients, random numbers and
noise.
"""

from . import blob as blob, decay as decay, gradient as gradient, instrument as instrument, noise as noise
from _typeshed import Incomplete
def __getattr__(name: str) -> Incomplete: ...
