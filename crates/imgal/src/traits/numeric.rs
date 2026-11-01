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

    /// Convert from this type to usize with potential precision loss.
    fn to_usize(self) -> usize;

    /// Convert from this type to f32 with potential precision loss.
    fn to_f32(self) -> f32;

    /// Convert from this type to f64 with potential precision loss.
    fn to_f64(self) -> f64;

    /// Convert from f32 to this type with potential precision loss.
    fn from_f32(value: f32) -> Self;

    /// Convert from f64 to this type with potential precision loss.
    fn from_f64(value: f64) -> Self;

    /// Convert from i32 to this type with potential precision loss.
    fn from_i32(value: i32) -> Self;
}

impl AsNumeric for usize {
    const MAX: Self = usize::MAX;
    const MIN: Self = usize::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as usize
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as usize
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as usize
    }
}

impl AsNumeric for u8 {
    const MAX: Self = u8::MAX;
    const MIN: Self = u8::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as u8
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as u8
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as u8
    }
}

impl AsNumeric for u16 {
    const MAX: Self = u16::MAX;
    const MIN: Self = u16::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as u16
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as u16
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as u16
    }
}

impl AsNumeric for u32 {
    const MAX: Self = u32::MAX;
    const MIN: Self = u32::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as u32
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as u32
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as u32
    }
}

impl AsNumeric for u64 {
    const MAX: Self = u64::MAX;
    const MIN: Self = u64::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as u64
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as u64
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as u64
    }
}

impl AsNumeric for i8 {
    const MAX: Self = i8::MAX;
    const MIN: Self = i8::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as i8
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as i8
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as i8
    }
}

impl AsNumeric for i16 {
    const MAX: Self = i16::MAX;
    const MIN: Self = i16::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as i16
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as i16
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as i16
    }
}

impl AsNumeric for i32 {
    const MAX: Self = i32::MAX;
    const MIN: Self = i32::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as i32
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as i32
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value
    }
}

impl AsNumeric for i64 {
    const MAX: Self = i64::MAX;
    const MIN: Self = i64::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as i64
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as i64
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as i64
    }
}

impl AsNumeric for f32 {
    const MAX: Self = f32::MAX;
    const MIN: Self = f32::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as f32
    }
}

impl AsNumeric for f64 {
    const MAX: Self = f64::MAX;
    const MIN: Self = f64::MIN;

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value as f64
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as f64
    }
}
