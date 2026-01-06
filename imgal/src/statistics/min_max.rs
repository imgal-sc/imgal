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
///
/// # Returns
///
/// * `Ok(T)`: The maximum value in the input data array.
/// * `Err(ImgalError)`: If the input data array is empty.
#[inline]
pub fn max<'a, T, A, D>(data: A) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let max = match view.first() {
        Some(av) => Zip::from(&view).fold(av, |acc, v| if v > acc { v } else { acc }),
        None => {
            return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" });
        }
    };

    Ok(max.clone())
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
///
/// # Returns
///
/// * `Ok(T)`: The minimum value in the input data array.
/// * `Err(ImgalError)`: If the input data array is empty.
#[inline]
pub fn min<'a, T, A, D>(data: A) -> Result<T, ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let min = match view.first() {
        Some(av) => Zip::from(&view).fold(av, |acc, v| if v < acc { v } else { acc }),
        None => {
            return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" });
        }
    };

    Ok(min.clone())
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
///
/// # Returns
///
/// * `Ok((T, T))`: A tuple containing the minimum and maximum values (_i.e._
///   (min, max)) in the given array.
/// * `Err(ImgalError)`: If the input data array is empty.
#[inline]
pub fn min_max<'a, T, A, D>(data: A) -> Result<(T, T), ImgalError>
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let mm = match view.first() {
        Some(av) => Zip::from(&view).fold((av, av), |acc, v| {
            (
                if v < acc.0 { v } else { acc.0 },
                if v > acc.1 { v } else { acc.1 },
            )
        }),
        None => return Err(ImgalError::InvalidParameterEmptyArray { param_name: "data" }),
    };

    Ok((mm.0.clone(), mm.1.clone()))
}
