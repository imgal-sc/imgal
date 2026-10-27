"""
The `imgal` (**IM**a**G**e **A**lgorithm **L**ibrary) crate is a fast and
open-source scientific image processing and algorithm library. This library
is directly inspired by [imagej-ops](https://github.com/imagej/imagej-ops/),
[SciJava Ops](https://github.com/scijava/scijava),
[ImgLib2](https://github.com/imglib/imglib2), and the ImageJ2 ecosystem.
`imgal` library aims to offer users access to fast and well documented
scientific image processing and analysis algorithms.

This project is still under active development and it's API is not stable.
"""

from . import colocalization as colocalization, copy as copy, distribution as distribution, filter as filter, gpu as gpu, image as image, integration as integration, kernel as kernel, overlay as overlay, parameter as parameter, phasor as phasor, simulation as simulation, spatial as spatial, statistics as statistics, threshold as threshold, transform as transform

__all__ = ["colocalization", "copy", "distribution", "filter", "gpu", "image", "integration", "kernel", "overlay", "parameter", "phasor", "simulation", "spatial", "statistics", "threshold", "transform"]
