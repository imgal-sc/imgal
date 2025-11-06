use pyo3::PyErr;
use pyo3::exceptions::{PyException, PyIndexError, PyValueError};

use imgal::error::ImgalError;

/// Map ImgalError types to Python exceptions.
pub fn map_imgal_error(err: ImgalError) -> PyErr {
    match err {
        ImgalError::InvalidArrayGeneric { msg } => PyException::new_err(format!("{}", msg)),
        ImgalError::InvalidArrayParameterValueEqual { param_name, value } => {
            PyValueError::new_err(format!(
                "Invalid array parameter value, the parameter {} can not equal {}.",
                param_name, value
            ))
        }
        ImgalError::InvalidArrayParameterValueGreater { param_name, value } => {
            PyValueError::new_err(format!(
                "Invalid array parameter value, the parameter {} can not be greater than {}.",
                param_name, value
            ))
        }
        ImgalError::InvalidArrayParameterValueLess { param_name, value } => {
            PyValueError::new_err(format!(
                "Invalid array parameter value, the parameter {} can not be less than {}.",
                param_name, value
            ))
        }
        ImgalError::InvalidAxis { axis_idx, dim_len } => PyIndexError::new_err(format!(
            "Axis {} is out of bounds for dimension length {}.",
            axis_idx, dim_len
        )),
        ImgalError::InvalidParameterValueOutsideRange {
            param_name,
            value,
            min,
            max,
        } => PyValueError::new_err(format!(
            "Invalid parameter value, the parameter {} must be a value between {} and {} but got {}.",
            param_name, min, max, value
        )),
        ImgalError::InvalidSum { expected, got } => PyValueError::new_err(format!(
            "Invalid sum, expected {} but got {}.",
            expected, got
        )),
        ImgalError::MismatchedArrayLengths {
            a_arr_len,
            b_arr_len,
        } => PyValueError::new_err(format!(
            "Mismatched array lengths, {} and {}, do not match.",
            a_arr_len, b_arr_len
        )),
        ImgalError::MismatchedArrayShapes { shape_a, shape_b } => PyValueError::new_err(format!(
            "Mismatched array shapes, {:?} and {:?}, do not match.",
            shape_a, shape_b
        )),
    }
}
