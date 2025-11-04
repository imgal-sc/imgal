use ndarray::ArrayViewD;

use crate::traits::numeric::AsNumeric;

/// Find the maximum value in an n-dimensional array.
///
/// # Description
///
/// This function iterates through all elements of an n-dimensional array to
/// determine the maximum value.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array view.
///
/// # Returns
///
/// * `T`: The maximum value in the input data array.
#[inline]
pub fn max<T>(data: ArrayViewD<T>) -> T
where
    T: AsNumeric,
{
    let m = data.iter().reduce(|acc, v| if v > acc { v } else { acc });

    *m.unwrap_or(&T::default())
}

/// Find the minimum value in an n-dimensional array.
///
/// # Description
///
/// This function iterates through all elements of an n-dimensional array to
/// determine the minimum value.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array view.
///
/// # Returns
///
/// * `T`: The minimum value in the input data array.
#[inline]
pub fn min<T>(data: ArrayViewD<T>) -> T
where
    T: AsNumeric,
{
    let m = data.iter().reduce(|acc, v| if v < acc { v } else { acc });

    *m.unwrap_or(&T::default())
}

/// Find the minimum and maximum values in an n-dimensional array.
///
/// # Description
///
/// This function iterates through all elements of an n-dimensional array to
/// determine the minimum and maximum values.
///
/// # Arguments
///
/// * `data`: The input n-dimensional array view.
///
/// # Returns
///
/// * `(T, T)`: A tuple containing the minimum and maximum values (_i.e._
///    (min, max)) in the given array. If the array is empty a minimum and
///    maximum value of 0 is returned in the tuple.
#[inline]
pub fn min_max<T>(data: ArrayViewD<T>) -> (T, T)
where
    T: AsNumeric,
{
    let mm = data.iter().fold(None, |acc, &v| {
        Some(match acc {
            None => (v, v),
            Some((min, max)) => (if v < min { v } else { min }, if v > max { v } else { max }),
        })
    });

    mm.unwrap_or_default()
}
