use ndarray::{ArrayBase, AsArray, Dimension, ViewRepr, Zip};

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
    let max = Zip::from(&view).fold(arbitrary_value, |acc, v| if v > acc { v } else { acc });

    max.clone()
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
    let max = Zip::from(&view).fold(arbitrary_value, |acc, v| if v < acc { v } else { acc });

    max.clone()
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
/// * `(T, T)`: A tuple containing the minimum and maximum values (_i.e._
///   (min, max)) in the given array.
#[inline]
pub fn min_max<'a, T, A, D>(data: A) -> (T, T)
where
    A: AsArray<'a, T, D>,
    D: Dimension,
    T: 'a + PartialOrd + Clone + Sync,
{
    let view: ArrayBase<ViewRepr<&'a T>, D> = data.into();
    let arbitrary_value = view.first().unwrap();
    let mm = Zip::from(&view).fold((arbitrary_value, arbitrary_value), |acc, v| {
        (
            if v < acc.0 { v } else { acc.0 },
            if v > acc.1 { v } else { acc.1 },
        )
    });

    (mm.0.clone(), mm.1.clone())
}
