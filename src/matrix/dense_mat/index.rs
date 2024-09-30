use std::ops::{Index, IndexMut};

use crate::matrix::mat_shape::MatShape;

use super::DenseMat;

impl<T> DenseMat<T> {
    #[inline]
    pub(crate) fn index_2d_to_1d(&self, idx_x: usize, idx_y: usize) -> usize {
        idx_x * self.ny + idx_y
    }
}

impl<T> Index<(usize, usize)> for DenseMat<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[self.index_2d_to_1d(index.0, index.1)]
    }
}

impl<T> IndexMut<(usize, usize)> for DenseMat<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.index_2d_to_1d(index.0, index.1);
        &mut self.data[idx]
    }
}

impl<T> MatShape for DenseMat<T> {
    fn nx(&self) -> usize {
        self.nx
    }
    fn ny(&self) -> usize {
        self.ny
    }
}
