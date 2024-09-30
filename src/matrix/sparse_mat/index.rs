use crate::matrix::mat_shape::MatShape;

impl<T> super::SparseMat<T> {
    pub fn get(&self, index: (usize, usize)) -> Option<&T> {
        assert!(
            index.0 < self.nx && index.1 < self.ny,
            "Index out of range, Mat shape is ({}, {}), but index is {index:?}",
            self.nx,
            self.ny
        );
        self.data.get(&index)
    }

    pub fn set(&mut self, index: (usize, usize), value: T) {
        assert!(
            index.0 < self.nx && index.1 < self.ny,
            "Index out of range, Mat shape is ({}, {}), but index is {index:?}",
            self.nx,
            self.ny
        );
        self.data.insert(index, value);
    }

    pub fn remove(&mut self, index: (usize, usize)) {
        assert!(
            index.0 < self.nx && index.1 < self.ny,
            "Index out of range, Mat shape is ({}, {}), but index is {index:?}",
            self.nx,
            self.ny
        );
        self.data.remove(&index);
    }
}

impl<T> MatShape for super::SparseMat<T> {
    fn nx(&self) -> usize {
        self.nx
    }
    fn ny(&self) -> usize {
        self.ny
    }
}
