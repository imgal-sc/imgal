use ndarray::ArrayView2;

use crate::traits::numeric::AsNumeric;

/// An immutable K-d tree for fast spatial queries for n-dimensional points.
///
/// The KD-tree itself does not *own* its source data but instead uses a view.
/// This design ensures that imgal's KD-trees are *immutable* once constructed
/// and are intended for lookups only. The `cloud` (*i.e.* the *n*-dimensional
/// point cloud) views points in *k* dimensions with shape `(p, k)`, where `p`
/// is the point and `k` is the dimension/axis of that point.
pub struct KDTree<'a, T> {
    pub cloud: ArrayView2<'a, T>,
    pub nodes: Vec<Node>,
    pub root: Option<usize>,
}

/// A K-d-tree node for an immutable K-d tree.
///
/// KD-trees are constructed with `Node`s. These `Nodes` are stored in a
/// `Vec<Node>` and the `left` and `right` fields store indices into the `Node`
/// vector. The axis the split occurs at is stored in `split_ax` and the index
/// into the source array is stored in the `point_index` field.
pub struct Node {
    pub split_ax: usize,
    pub point_index: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl<'a, T> KDTree<'a, T>
where
    T: AsNumeric,
{
    /// Creates a new K-d tree from an *n*-dimensional point cloud.
    pub fn build(cloud: ArrayView2<'a, T>) -> Self {
        let mut tree = Self {
            cloud,
            nodes: Vec::new(),
            root: None,
        };
        let total_points = cloud.dim().1;
        let indices: Vec<usize> = (0..total_points).collect();
        tree.root = tree.recurse(&indices, 0);

        tree
    }

    fn recurse(&mut self, indices: &[usize], depth: usize) -> Option<usize> {
        todo!("Implement the build recusion function.")
    }
}

impl Node {
    /// Creates a new K-d tree node.
    pub fn new(
        split_ax: usize,
        point_index: usize,
        left: Option<usize>,
        right: Option<usize>,
    ) -> Self {
        Self {
            split_ax,
            point_index,
            left,
            right,
        }
    }
}
