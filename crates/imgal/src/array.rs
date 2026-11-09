//! Array module, contains the ImgArray struct and helper functions.

use crate::prelude::*;

pub enum MemoryOrder {
    C,
    F
}

pub struct ImgArray<T>
where
    T: AsNumeric,
{
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub stride: usize,
    pub order: MemoryOrder,
}

impl<T> ImgArray<T>
where
    T: AsNumeric,
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
