use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};

pub trait ToFloat64:
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
    fn to_f64(self) -> f64;
}

// unsigned to f64, there is precision loss with u64
impl ToFloat64 for u8 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToFloat64 for u16 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToFloat64 for u32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToFloat64 for u64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// signed to f64, there is precision loss with i64
impl ToFloat64 for i8 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToFloat64 for i16 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToFloat64 for i32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToFloat64 for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// float to f64, no precision loss
impl ToFloat64 for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl ToFloat64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}
