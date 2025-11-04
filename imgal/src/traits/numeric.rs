use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};

/// Trait for numeric types that can be converted to and from f64 with potential
/// precision loss.
pub trait AsNumeric:
    Copy
    + Add<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + MulAssign
    + Sum
    + Debug
    + Default
    + PartialOrd
    + Send
    + Sync
{
    /// Convert from this type to f64 with potential precision loss with i64
    /// and u64.
    fn to_f64(self) -> f64;

    /// Convert from f64 to this type with potential precision loss.
    fn from_f64(value: f64) -> Self;
}

impl AsNumeric for u8 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as u8
    }
}

impl AsNumeric for u16 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as u16
    }
}

impl AsNumeric for u32 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as u32
    }
}

impl AsNumeric for u64 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as u64
    }
}

impl AsNumeric for i8 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as i8
    }
}

impl AsNumeric for i16 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as i16
    }
}

impl AsNumeric for i32 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as i32
    }
}

impl AsNumeric for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as i64
    }
}

impl AsNumeric for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl AsNumeric for f64 {
    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }
}
