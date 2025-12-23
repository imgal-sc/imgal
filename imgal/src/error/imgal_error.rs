use std::error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ImgalError {
    InvalidAxis {
        axis_idx: usize,
        dim_len: usize,
    },
    InvalidAxisValueGreaterEqual {
        arr_name: &'static str,
        axis_idx: usize,
        value: usize,
    },
    InvalidGeneric {
        msg: &'static str,
    },
    InvalidParameterEmptyArray {
        param_name: &'static str,
    },
    InvalidParameterValueEqual {
        param_name: &'static str,
        value: usize,
    },
    InvalidParameterValueGreater {
        param_name: &'static str,
        value: usize,
    },
    InvalidParameterValueLess {
        param_name: &'static str,
        value: usize,
    },
    InvalidParameterValueOutsideRange {
        param_name: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    InvalidSum {
        expected: f64,
        got: f64,
    },
    MismatchedArrayLengths {
        a_arr_name: &'static str,
        a_arr_len: usize,
        b_arr_name: &'static str,
        b_arr_len: usize,
    },
    MismatchedArrayShapes {
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    },
}

impl fmt::Display for ImgalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImgalError::InvalidAxis { axis_idx, dim_len } => {
                write!(
                    f,
                    "Invalid axis, axis {} is out of bounds for dimension length {}.",
                    axis_idx, dim_len
                )
            }
            ImgalError::InvalidAxisValueGreaterEqual {
                arr_name,
                axis_idx,
                value,
            } => {
                write!(
                    f,
                    "Invalid axis value, axis {} of \"{}\" can not be greater than or equal to {}.",
                    axis_idx, arr_name, value
                )
            }
            ImgalError::InvalidGeneric { msg } => {
                write!(f, "{}", msg)
            }
            ImgalError::InvalidParameterEmptyArray { param_name } => {
                write!(
                    f,
                    "Invalid array parameter, the array \"{}\" can not be empty.",
                    param_name
                )
            }
            ImgalError::InvalidParameterValueEqual { param_name, value } => {
                write!(
                    f,
                    "Invalid parameter value, the parameter {} can not equal {}.",
                    param_name, value
                )
            }
            ImgalError::InvalidParameterValueGreater { param_name, value } => {
                write!(
                    f,
                    "Invalid parameter value, the parameter {} can not be greater than {}.",
                    param_name, value
                )
            }
            ImgalError::InvalidParameterValueLess { param_name, value } => {
                write!(
                    f,
                    "Invalid parameter value, the parameter {} can not be less than {}.",
                    param_name, value
                )
            }
            ImgalError::InvalidParameterValueOutsideRange {
                param_name,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "Invalid parameter value, the parameter {} must be a value between {} and {} but got {}.",
                    param_name, min, max, value
                )
            }
            ImgalError::InvalidSum { expected, got } => {
                write!(f, "Invalid sum, expected {} but got {}.", expected, got)
            }
            ImgalError::MismatchedArrayLengths {
                a_arr_name,
                a_arr_len,
                b_arr_name,
                b_arr_len,
            } => {
                write!(
                    f,
                    "Mismatched array lengths, \"{}\" of length {} and \"{}\" of length {} do not match.",
                    a_arr_name, a_arr_len, b_arr_name, b_arr_len
                )
            }
            ImgalError::MismatchedArrayShapes { shape_a, shape_b } => {
                write!(
                    f,
                    "Mismatched array shapes, {:?} and {:?}, do not match.",
                    shape_a, shape_b
                )
            }
        }
    }
}

impl error::Error for ImgalError {}
