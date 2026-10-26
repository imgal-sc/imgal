"""
Convex hull, geometry, halfplane/halfspace and spatial data structures.

This module provides various functions for computing 2D and 3D convex hulls,
geometric operations, and spatial data structures (*e.g* Kd-Trees and
regions of interest (ROIs)).
"""

from . import convex_hull as convex_hull, geometry as geometry, halfplane as halfplane, halfspace as halfspace, roi as roi
from _typeshed import Incomplete
def __getattr__(name: str) -> Incomplete: ...
