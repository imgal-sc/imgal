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
    // maximum and minimum values for numeric type
    const MAX: Self;
    const MIN: Self;

    /// Convert from this type to f64 with potential precision loss.
    fn to_usize(self) -> usize;

    /// Convert from this type to f64 with potential precision loss.
    fn to_f64(self) -> f64;

    /// Convert from f64 to this type with potential precision loss.
    fn from_f64(value: f64) -> Self;

    /// Convert from i32 to this type with potential precision loss.
    fn from_i32(value: i32) -> Self;
}

impl AsNumeric for usize {
    const MAX: Self = usize::MAX;
    const MIN: Self = usize::MIN;

    fn to_usize(self) -> usize {
        self
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as usize
    }

    fn from_i32(value: i32) -> Self {
        value as usize
    }
}

impl AsNumeric for u8 {
    const MAX: Self = u8::MAX;
    const MIN: Self = u8::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as u8
    }

    fn from_i32(value: i32) -> Self {
        value as u8
    }
}

impl AsNumeric for u16 {
    const MAX: Self = u16::MAX;
    const MIN: Self = u16::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as u16
    }

    fn from_i32(value: i32) -> Self {
        value as u16
    }
}

impl AsNumeric for u32 {
    const MAX: Self = u32::MAX;
    const MIN: Self = u32::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as u32
    }

    fn from_i32(value: i32) -> Self {
        value as u32
    }
}

impl AsNumeric for u64 {
    const MAX: Self = u64::MAX;
    const MIN: Self = u64::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as u64
    }

    fn from_i32(value: i32) -> Self {
        value as u64
    }
}

impl AsNumeric for i8 {
    const MAX: Self = i8::MAX;
    const MIN: Self = i8::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as i8
    }

    fn from_i32(value: i32) -> Self {
        value as i8
    }
}

impl AsNumeric for i16 {
    const MAX: Self = i16::MAX;
    const MIN: Self = i16::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as i16
    }

    fn from_i32(value: i32) -> Self {
        value as i16
    }
}

impl AsNumeric for i32 {
    const MAX: Self = i32::MAX;
    const MIN: Self = i32::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as i32
    }

    fn from_i32(value: i32) -> Self {
        value
    }
}

impl AsNumeric for i64 {
    const MAX: Self = i64::MAX;
    const MIN: Self = i64::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as i64
    }

    fn from_i32(value: i32) -> Self {
        value as i64
    }
}

impl AsNumeric for f32 {
    const MAX: Self = f32::MAX;
    const MIN: Self = f32::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn from_i32(value: i32) -> Self {
        value as f32
    }
}

impl AsNumeric for f64 {
    const MAX: Self = f64::MAX;
    const MIN: Self = f64::MIN;

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn from_i32(value: i32) -> Self {
        value as f64
    }
}
