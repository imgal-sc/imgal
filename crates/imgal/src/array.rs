//! Array module, contains the ImgArray struct and helper functions.

use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::sync::atomic::*;

pub trait Data: Debug + Send + Sized + Sync {
    /// The default data value.
    fn zero() -> Self;
}

impl Data for bool {
    #[inline]
    fn zero() -> Self {
        false
    }
}

impl Data for usize {
    #[inline]
    fn zero() -> Self {
        0_usize
    }
}

impl Data for u8 {
    #[inline]
    fn zero() -> Self {
        0_u8
    }
}

impl Data for u16 {
    #[inline]
    fn zero() -> Self {
        0_u16
    }
}

impl Data for u32 {
    #[inline]
    fn zero() -> Self {
        0_u32
    }
}

impl Data for u64 {
    #[inline]
    fn zero() -> Self {
        0_u64
    }
}

impl Data for isize {
    #[inline]
    fn zero() -> Self {
        0_isize
    }
}

impl Data for i8 {
    #[inline]
    fn zero() -> Self {
        0_i8
    }
}

impl Data for i16 {
    #[inline]
    fn zero() -> Self {
        0_i16
    }
}

impl Data for i32 {
    #[inline]
    fn zero() -> Self {
        0_i32
    }
}

impl Data for i64 {
    #[inline]
    fn zero() -> Self {
        0_i64
    }
}

impl Data for f32 {
    #[inline]
    fn zero() -> Self {
        0.0_f32
    }
}

impl Data for f64 {
    #[inline]
    fn zero() -> Self {
        0.0_f64
    }
}

impl Data for AtomicBool {
    #[inline]
    fn zero() -> Self {
        AtomicBool::new(false)
    }
}

impl Data for AtomicUsize {
    #[inline]
    fn zero() -> Self {
        AtomicUsize::new(0)
    }
}

impl Data for AtomicU8 {
    #[inline]
    fn zero() -> Self {
        AtomicU8::new(0)
    }
}

impl Data for AtomicU16 {
    #[inline]
    fn zero() -> Self {
        AtomicU16::new(0)
    }
}

impl Data for AtomicU32 {
    #[inline]
    fn zero() -> Self {
        AtomicU32::new(0)
    }
}

impl Data for AtomicU64 {
    #[inline]
    fn zero() -> Self {
        AtomicU64::new(0)
    }
}

impl Data for AtomicIsize {
    #[inline]
    fn zero() -> Self {
        AtomicIsize::new(0)
    }
}

impl Data for AtomicI8 {
    #[inline]
    fn zero() -> Self {
        AtomicI8::new(0)
    }
}

impl Data for AtomicI16 {
    #[inline]
    fn zero() -> Self {
        AtomicI16::new(0)
    }
}

impl Data for AtomicI32 {
    #[inline]
    fn zero() -> Self {
        AtomicI32::new(0)
    }
}

impl Data for AtomicI64 {
    #[inline]
    fn zero() -> Self {
        AtomicI64::new(0)
    }
}

/// TODO
#[derive(Clone, Copy, Debug)]
pub enum MemoryOrder {
    C,
    F,
}

/// TODO
#[derive(Debug)]
pub struct ImgArray<T>
where
    T: Data,
{
    data: Vec<T>,
    shape: Vec<usize>,
    size: usize,
    offset: usize,
    stride: usize,
    order: MemoryOrder,
}

impl<T> ImgArray<T>
where
    T: Data,
{
    /// Creare a new ImgArray from raw data and shape.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let size = data.len();
        Self {
            data,
            shape,
            size,
            offset: 0,
            stride: 1,
            order: MemoryOrder::C,
        }
    }

    /// Create an ImgArray with default values.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product::<usize>();
        let data = (0..size).map(|_| T::zero()).collect::<Vec<T>>();
        Self {
            data,
            shape,
            size,
            offset: 0,
            stride: 1,
            order: MemoryOrder::C,
        }
    }
}

impl<T> ImgArray<T>
where
    T: Data,
{
    /// TODO
    fn offset(&self) -> usize {
        self.offset
    }

    /// TODO
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// TODO
    fn size(&self) -> usize {
        self.size
    }
}
