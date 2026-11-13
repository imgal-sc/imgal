//! Array module, contains the ImgArray struct and helper functions.

use std::fmt::Debug;
use std::sync::atomic::*;

pub trait Data: Debug + Send + Sized + Sync {}

macro_rules! impl_data {
    ($($t:ty), * $(,)?) => {
        $(impl Data for $t {})*
    };
}

impl_data!(
    bool,
    usize,
    u8,
    u16,
    u32,
    u64,
    isize,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    AtomicBool,
    AtomicUsize,
    AtomicU8,
    AtomicU16,
    AtomicU32,
    AtomicU64,
    AtomicIsize,
    AtomicI8,
    AtomicI16,
    AtomicI32,
    AtomicI64,
);

/// TODO
#[derive(Debug)]
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
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub stride: usize,
    pub order: MemoryOrder,
}

impl<T> ImgArray<T>
where
    T: Data,
{
    /// Creare a new ImgArray from raw data and shape.
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            stride: 1,
            order: MemoryOrder::C,
        }
    }
}
