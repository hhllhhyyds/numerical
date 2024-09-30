use std::collections::BTreeMap;

use super::SparseMat;

impl<T> SparseMat<T> {
    pub fn new(nx: usize, ny: usize, elements: impl Iterator<Item = (usize, usize, T)>) -> Self {
        let mut mat = Self {
            nx,
            ny,
            data: BTreeMap::default(),
        };

        elements.for_each(|elem| {
            let (ix, iy, val) = elem;
            mat.set((ix, iy), val);
        });

        mat
    }

    pub fn full_from_vec(nx: usize, ny: usize, elements: Vec<T>) -> Self {
        assert!(nx * ny == elements.len());
        Self::new(
            nx,
            ny,
            elements
                .into_iter()
                .enumerate()
                .map(|(i, val)| (i / ny, i % ny, val)),
        )
    }
}
