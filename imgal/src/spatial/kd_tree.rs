use ndarray::{ArrayViewD};

use crate::traits::numeric::AsNumeric;

/// A KD-tree.
///
/// The KD-tree itself does not *own* its source data but instead uses a view.
/// This design ensures that imgal's KD-trees are *immutable* once constructed
/// and are intended for lookups only.
pub struct KDTree<'a, T>{
    pub source: ArrayViewD<'a, T>,
    pub nodes: Vec<Node>,
    pub root: Option<usize>,
}

/// A KD-tree node.
///
/// KD-trees are constructed with `Node`s. These `Nodes` are stored in a
/// `Vec<Node>` and the `left` and `right` fields store indices into the `Node`
/// vector.
pub struct Node {
    pub split_dim: usize,
    pub point_index: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

