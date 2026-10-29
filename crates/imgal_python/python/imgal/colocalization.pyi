"""
Colocalization analysis on 2D and 3D data.

This module provides colocalization analysis functions for both 2D and 3D
image data.
"""

from _typeshed import Incomplete
from typing import Any

def pearson_roi_coloc(data_a: Any, data_b: Any, rois: dict[int, Incomplete], threads: int | None = None) -> dict[int, float]:
    """
    Compute the Pearson correlation coefficient between two n-dimensional images
    and a ROI map.

    Computes the Pearson correlation coefficient, a measure of linear
    correlation between two sets of n-dimensional images and a ROI map. This
    function iterates through each ROI in the map and computes the correlation
    coefficient. Returning a `HashMap` of Pearson correlation coefficient values
    and ROI label IDs.

    Args:
        data_a: The first n-dimensional image for Pearson colocalization
            analysis.
        data_b: the second n-dimensional image for Pearson colocalization
            analysis.
        rois: A map of point clouds representing Regions of Interest (ROIs).
            The individual ROIs must have the same dimensionality as the input
            data.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        A `HashMap` where the keys are the ROI label IDs and values are the
        Pearson correlation coefficients for each ROI respectively.

    Errors:
        If `len(data_a) != len(data_b)`. If `len(data_a)` or `len(data_b)` or
        `len(data_b)` is <= 2.
    """

def saca_2d(data_a: Any, data_b: Any, threshold_a: float, threshold_b: float, threads: int | None = None) -> Incomplete:
    """
    Compute 2D colocalization strength with Spatially Adaptive Colocalization
    Analysis (SACA).

    Computes a pixel-wise *z-score* indicating colocalization and
    anti-colocalization strength on 2D input images using the Spatially Adaptive
    Colocalization Analysis (SACA) framework. Per pixel SACA utilizes a
    propagation and separation strategy to adaptively expand a weighted
    circular kernel that defines the pixel of consideration's neighborhood.
    The pixels within the neighborhood are assigned weights based on their
    distance from the center pixel (decreasing with distance), ranked and their
    colocalization coefficient computed using Kendall's Tau-b rank correlation.

    Args:
        data_a: The 2D input image corresponding to the first channel.
        data_b: The 2D input image corresponding to the second channel.
        threshold_a: Pixel intensity threshold value for `data_a`. Pixels below
            this value are given a weight of `0.0` if the pixel is in the
            circular neighborhood.
        threshold_b: Pixel intensity threshold value for `data_b`. Pixels below
            this value are given a weight of `0.0` if the pixel is in the
            circular neighborhood.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The pixel-wise *z-score* indicating colocalization or
        anti-colocalization by its sign and the degree or strength of the
        relationship through its absolute values.

    Errors:
        If `data_a.shape != data_b.shape`.

    Reference:
        <https://doi.org/10.1109/TIP.2019.2909194>
    """

def saca_3d(data_a: Any, data_b: Any, threshold_a: float, threshold_b: float, threads: int | None = None) -> Incomplete:
    """
    Compute 3D colocalization strength with Spatially Adaptive Colocalization
    Analysis (SACA).

    Computes a pixel-wise *z-score* indicating colocalization and
    anti-colocalization strength on 3D input images using the Spatially Adaptive
    Colocalization Analysis (SACA) framework. Per pixel SACA utilizes a
    propagation and separation strategy to adaptively expand a weighted
    spherical kernel that defines the pixel of consideration's neighborhood.
    The pixels within the neighborhood are assigned weights based on their
    distance from the center pixel (decreasing with distance), ranked and
    their colocalization coefficient computed using Kendall's Tau-b rank
    correlation.

    Args:
        data_a: The 3D input image corresponding to the first channel.
        data_b: The 3D input image corresponding to the second channel.
        threshold_a: Pixel intensity threshold value for `data_a`. Pixels below
            this value are given a weight of `0.0` if the pixel is in the
            circular neighborhood.
        threshold_b: Pixel intensity threshold value for `data_b`. Pixels below
            this value are given a weight of `0.0` if the pixel is in the
            circular neighborhood.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The pixel-wise *z-score* indicating colocalization or
        anti-colocalization by its sign and the degree or strength of the
        relationship through its absolute values.

    Errors:
        If `data_a.shape != data_b.shape`.

    Reference:
        <https://doi.org/10.1109/TIP.2019.2909194>
    """

def saca_significance_mask(data: Any, alpha: float | None = None, threads: int | None = None) -> Incomplete:
    """
    Create a significant pixel mask from a pixel-wise *z-score* array.

    Creates a boolean image representing significant pixels (*i.e.* the mask) by
    applying Bonferroni correction to adjust for multiple comparisons.

    Args:
        data: The pixel-wise *z-score* indicating colocalization or
            anti-colocalization strength.
        alpha: The significance level representing the maximum type I error
            (*i.e.* false positive error) allowed. If `None` then
            `alpha = 0.05`.
        threads: The requested number of threads to use for parallel execution.
            If `None` or `1` sequential execution is used. If `0`, then the
            maximum available parallelism is used. Thread counts are clamped to
            the systems maximum.

    Returns:
        The significant pixel mask where `true` pixels represent significant
        *z-score* values.

    Reference:
        <https://doi.org/10.1109/TIP.2019.2909194>
    """

def __getattr__(name: str) -> Incomplete: ...
