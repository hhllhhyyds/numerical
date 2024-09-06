use std::collections::BTreeMap;

use crate::FloatCore;

use crate::mat_traits::MatShape;

#[derive(Clone, Debug, Default)]
pub struct SparseMat<T> {
    nx: usize,
    ny: usize,
    data: BTreeMap<(usize, usize), T>,
}

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

    pub fn get(&mut self, index: (usize, usize)) -> Option<&T> {
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

impl<T: FloatCore> SparseMat<T> {
    pub fn mul_vec(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());

        let mut v = vec![T::zero(); self.ny];
        for (&(ix, iy), &val) in self.data.iter() {
            v[iy] = v[iy] + val * b[ix];
        }

        v
    }

    pub fn jacobi_iterate(x_k: &mut [T], b: &[T], diagonal: &[T], l_plus_u: &Self) {
        assert!(l_plus_u.is_square());
        let n = l_plus_u.nx;
        assert!(x_k.len() == n);
        assert!(b.len() == n);
        assert!(diagonal.len() == n);

        let x = l_plus_u.mul_vec(x_k);
        (0..n).for_each(|i| x_k[i] = (b[i] - x[i]) / diagonal[i]);
    }
}

impl<T> MatShape for SparseMat<T> {
    fn shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }
}

#[cfg(test)]
mod tests {
    use approx::AbsDiffEq;

    use super::*;

    #[test]
    fn test_jacobi_iterate_0() {
        let n = 100_000;
        let elems_0 = (0..n).map(|i| (i, n - i - 1, 0.5));
        let elems_1 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
        let elems_2 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));

        let elems = elems_0.chain(elems_1).chain(elems_2);
        let mat = SparseMat::new(n, n, elems);

        let b = {
            let mut b = vec![1.5; n];
            b[0] = 2.5;
            b[n - 1] = 2.5;
            b[n / 2 - 1] = 1.0;
            b[n / 2] = 1.0;
            b
        };

        let mut x = vec![0.0; n];

        let diag = vec![3.0; n];

        let start = std::time::Instant::now();
        for _ in 0..50 {
            SparseMat::jacobi_iterate(&mut x, &b, &diag, &mat);
        }
        println!(
            "n = {n}, time used in 50 jacobi iteration = {} ms",
            (std::time::Instant::now() - start).as_millis()
        );

        assert!(x.iter().all(|y| y.abs_diff_eq(&1.0, 1e-6)));
    }
}
