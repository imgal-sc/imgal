use std::cmp::Ordering;

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
/// vector. The axis the split occurs at is stored in `split_axis` and the index
/// into the source array is stored in the `point_index` field.
pub struct Node {
    pub split_axis: usize,
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

    /// Recursively build the K-d tree.
    fn recurse(&mut self, indices: &[usize], depth: usize) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }
        let ndims = self.cloud.dim().1;
        let split_axis = depth % ndims;
        let mut inds_sorted = indices.to_vec();
        inds_sorted.sort_by(|&a, &b| {
            self.cloud[[a, split_axis]]
                .partial_cmp(&self.cloud[[b, split_axis]])
                .unwrap_or(Ordering::Less)
        });
        let median = inds_sorted.len() / 2;
        let point_index = inds_sorted[median];
        // construct the left and right sub trees
        let left = self.recurse(&inds_sorted[..median], depth + 1);
        let right = self.recurse(&inds_sorted[..median], depth + 1);
        // create a new Node and return this Node's index
        let node_index = self.nodes.len();
        self.nodes
            .push(Node::new(split_axis, point_index, left, right));

        Some(node_index)
    }
}

impl Node {
    /// Creates a new K-d tree node.
    pub fn new(
        split_axis: usize,
        point_index: usize,
        left: Option<usize>,
        right: Option<usize>,
    ) -> Self {
        Self {
            split_axis,
            point_index,
            left,
            right,
        }
    }
}
