use ndarray::{ArrayBase, ArrayViewD, AsArray, Dimension, ViewRepr, Zip};

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
pub fn max<'a, T, A, D>(data: A) -> T
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let arbitrary_value = view.first().unwrap();
    let max = Zip::from(&view).par_fold(
        || arbitrary_value,
        |acc, v| if v > acc { v } else { acc },
        |acc, v| if v > acc { v } else { acc }
    );
    max.clone()
}

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
pub fn max_sequential<T>(data: ArrayViewD<T>) -> T
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
pub fn min<'a, T, A, D>(data: A) -> T
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let arbitrary_value = view.first().unwrap();
    let max = Zip::from(&view).par_fold(
        || arbitrary_value,
        |acc, v| if v < acc { v } else { acc },
        |acc, v| if v < acc { v } else { acc }
    );
    max.clone()
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
pub fn min_max<'a, T, A, D>(data: A) -> (T, T)
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let arbitrary_value = view.first().unwrap();
    let mm = Zip::from(&view).par_fold(
        || (arbitrary_value, arbitrary_value),
        |acc, v| (if v < acc.0 { v } else { acc.0}, if v > acc.1 { v } else { acc.1 }),
        |acc, v| (if v.0 < acc.0 { v.0 } else { acc.0 },
                   if v.1 > acc.1 { v.1 } else { acc.1 })
    );
    (mm.0.clone(), mm.1.clone())
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
pub fn min_max_sequential<T>(data: ArrayViewD<T>) -> (T, T)
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
