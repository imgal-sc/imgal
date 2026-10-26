"""
*n*-dimensional region of interest (ROI) functions.

This module provides functions for creating *n*-dimensional ROIs and
querying ROI bounded data.
"""

from _typeshed import Incomplete
from typing import Any

def roi_cloud_map(labels: Any, threads: int | None = None) -> dict[int, Incomplete]:
    """
    Create a ROI point cloud map from an n-dimensional label image.

    Creates a region of interest (ROI) "cloud" map from an n-dimensional label
    image. For a given input image each label is converted into a 2D array
    representing a point cloud with shape `(p, D)`, where `p` and `D` are the
    number of points and dimensions respectively. Each label's point cloud is
    stored with it's associated key (*i.e.* label ID) in the output `HashMap`.

    Args:
        labels: The n-dimensional label image.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A ROI `HashMap` where the keys are the ROI label IDs and values are the
        ROI point clouds.
    """

def roi_data_map(data: Any, labels: Incomplete, threads: int | None = None) -> dict[int, Any]:
    """
    Create a ROI data map from n-dimensional data and a label image.

    Creates a region of interest (ROI) "data" map from input n-dimensional data
    and label images. For a given `data` and `labels` image pair, each
    coordinate within every label in the label image is used to query the
    image data. Each label's associated raw data is stored as a 1D array with
    the label's key (*i.e.* label ID) in the output `HashMap`.

    Args:
        data: The input n-dimensional image data.
        labels: The corresponding n-dimensional label image for `data`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A ROI `HashMap` where the keys are the ROI label IDs and the values are
        1D arrays containing raw values from the ROI.
    """

def __getattr__(name: str) -> Incomplete: ...
