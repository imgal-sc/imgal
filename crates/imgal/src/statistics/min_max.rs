use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr, Zip};

use crate::error::ImgalError;

/// Find the maximum value in an n-dimensional array.
///
/// # Description
///
/// Iterates through all elements of an n-dimensional array to determine the
/// maximum value.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array view.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(T)`: The maximum value in the input data array.
/// * `Err(ImgalError)`: If the input data array is empty.
#[inline]
pub fn max<'a, T, A, D>(data: A, parallel: bool) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    if parallel {
        let max = match data.first() {
            Some(av) => Zip::from(&data).par_fold(
                || av,
                |acc, v| if v > acc { v } else { acc },
                |acc, v| if v > acc { v } else { acc },
            ),
            None => return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" }),
        };
        Ok(max.clone())
    } else {
        let max = match data.first() {
            Some(av) => Zip::from(&data).fold(av, |acc, v| if v > acc { v } else { acc }),
            None => {
                return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" });
            }
        };
        Ok(max.clone())
    }
}

/// Find the minimum value in an n-dimensional array.
///
/// # Description
///
/// Iterates through all elements of an n-dimensional array to determine the
/// minimum value.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array view.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok(T)`: The minimum value in the input data array.
/// * `Err(ImgalError)`: If the input data array is empty.
#[inline]
pub fn min<'a, T, A, D>(data: A, parallel: bool) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    if parallel {
        let min = match data.first() {
            Some(av) => Zip::from(&data).par_fold(
                || av,
                |acc, v| if v < acc { v } else { acc },
                |acc, v| if v < acc { v } else { acc },
            ),
            None => return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" }),
        };
        Ok(min.clone())
    } else {
        let min = match data.first() {
            Some(av) => Zip::from(&data).fold(av, |acc, v| if v < acc { v } else { acc }),
            None => {
                return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" });
            }
        };
        Ok(min.clone())
    }
}

/// Find the minimum and maximum values in an n-dimensional array.
///
/// # Description
///
/// Iterates through all elements of an n-dimensional array to determine the
/// minimum and maximum values.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array view.
/// * `parallel`: If `true`, parallel computation is used across multiple
///   threads. If `false`, sequential single-threaded computation is used.
///
/// # Returns
///
/// * `Ok((T, T))`: A tuple containing the minimum and maximum values (_i.e._
///   (min, max)) in the given array.
/// * `Err(ImgalError)`: If the input data array is empty.
#[inline]
pub fn min_max<'a, T, A, D>(data: A, parallel: bool) -> Result<(T, T), ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let data: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    if parallel {
        let mm = match data.first() {
            Some(av) => Zip::from(&data).par_fold(
                || (av, av),
                |acc, v| {
                    (
                        if v < acc.0 { v } else { acc.0 },
                        if v > acc.1 { v } else { acc.1 },
                    )
                },
                |acc, v| {
                    (
                        if v.0 < acc.0 { v.0 } else { acc.0 },
                        if v.1 > acc.1 { v.1 } else { acc.1 },
                    )
                },
            ),
            None => return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" }),
        };
        Ok((mm.0.clone(), mm.1.clone()))
    } else {
        let mm = match data.first() {
            Some(av) => Zip::from(&data).fold((av, av), |acc, v| {
                (
                    if v < acc.0 { v } else { acc.0 },
                    if v > acc.1 { v } else { acc.1 },
                )
            }),
            None => return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" }),
        };
        Ok((mm.0.clone(), mm.1.clone()))
    }
}
