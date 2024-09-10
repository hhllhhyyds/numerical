use std::collections::BTreeMap;

use crate::FloatCore;

use crate::mat_traits::{MatOps, MatShape};

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

impl<T: Clone> SparseMat<T> {
    pub fn transpose(&self) -> Self {
        Self::new(
            self.ny,
            self.nx,
            self.data
                .iter()
                .map(|((ix, iy), val)| (*iy, *ix, val.clone())),
        )
    }
}

impl<T: FloatCore> SparseMat<T> {
    /// Transpose is needed before run gauss seidel iteration,
    /// because this method rely on the sorted BTreeMap
    pub fn gauss_seidel_iterate(&self, x_k: &mut [T], b: &[T]) {
        assert!(self.is_square());
        let n = self.nx();
        assert!(x_k.len() == n);
        assert!(b.len() == n);

        let mut iter = self.data.iter();
        let mut elem = iter.next();
        for ix_now in 0..n {
            let mut sum = T::zero();
            while let Some((&(ix, iy), &val)) = elem {
                if ix == ix_now {
                    if ix != iy {
                        sum = sum + val * x_k[iy];
                    }
                } else {
                    break;
                }
                elem = iter.next();
            }
            x_k[ix_now] = (b[ix_now] - sum) / *self.get((ix_now, ix_now)).unwrap_or(&T::zero());
        }
    }

    /// Transpose is needed before run gauss seidel iteration,
    /// because this method rely on the sorted BTreeMap
    pub fn sor_gauss_seidel_iterate(&self, x_k: &mut [T], b: &[T], omega: T) {
        assert!(self.is_square());
        let n = self.nx();
        assert!(x_k.len() == n);
        assert!(b.len() == n);

        let mut iter = self.data.iter();
        let mut elem = iter.next();
        for ix_now in 0..n {
            let mut sum = T::zero();
            while let Some((&(ix, iy), &val)) = elem {
                if ix == ix_now {
                    if ix != iy {
                        sum = sum + val * x_k[iy];
                    }
                } else {
                    break;
                }
                elem = iter.next();
            }
            x_k[ix_now] = (T::one() - omega) * x_k[ix_now]
                + omega * (b[ix_now] - sum) / *self.get((ix_now, ix_now)).unwrap_or(&T::zero());
        }
    }
}

impl<T> MatShape for SparseMat<T> {
    fn shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }
}

impl<T: FloatCore> MatOps<T> for SparseMat<T> {
    fn mul_vec(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());

        let mut v = vec![T::zero(); self.ny];
        for (&(ix, iy), &val) in self.data.iter() {
            v[iy] = v[iy] + val * b[ix];
        }

        v
    }
}

#[cfg(test)]
mod tests {
    use approx::AbsDiffEq;

    use super::*;

    #[test]
    #[ignore = "slow, run by hand"]
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

    #[test]
    fn test_jacobi_iterate_1() {
        let mat = SparseMat::new(2, 2, [(0_usize, 1_usize, 1.0_f64), (1, 0, 1.0)].into_iter());
        let diagonal = vec![3.0, 2.0];
        let mut x = vec![0.0; 2];
        let b = [5.0; 2];

        (0..20).for_each(|_| {
            SparseMat::jacobi_iterate(&mut x, &b, &diagonal, &mat);
        });

        (0..x.len()).for_each(|i| assert!(x[i].abs_diff_eq(&[1.0, 2.0][i], 1e-7)))
    }

    #[test]
    fn test_jacobi_iterate_2() {
        let n = 100;
        let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
        let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));

        let elems = elems_0.chain(elems_1);
        let mat = SparseMat::new(n, n, elems.clone());

        let b = {
            let mut b = vec![1.; n];
            b[0] = 2.;
            b[n - 1] = 2.;
            b
        };

        let mut x = vec![0.0; n];

        let diag = vec![3.0; n];

        let start = std::time::Instant::now();

        let mut iter_count = 0;
        while x
            .iter()
            .map(|val| (val - 1.).abs())
            .reduce(f64::max)
            .unwrap()
            >= 1e-6
        {
            iter_count += 1;
            SparseMat::jacobi_iterate(&mut x, &b, &diag, &mat);
        }

        println!(
            "n = {n}, time used in {iter_count} jacobi iteration = {} ms",
            (std::time::Instant::now() - start).as_millis()
        );

        let full_mat = SparseMat::new(n, n, elems.chain((0..n).map(|i| (i, i, 3.0))));
        let y = full_mat.mul_vec(&x);
        println!(
            "backward error = {:e}",
            y.iter()
                .zip(b.iter())
                .map(|(m, n)| (m - n).abs())
                .reduce(f64::max)
                .unwrap()
        );

        assert!(x.iter().all(|y| y.abs_diff_eq(&1.0, 1e-6)));
    }

    #[test]
    fn test_jacobi_iterate_3() {
        let n = 100;
        let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, 1.0));
        let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, 1.0));

        let elems = elems_0.chain(elems_1);
        let mat = SparseMat::new(n, n, elems.clone());

        let b = {
            let mut b = vec![0.; n];
            b[0] = 1.;
            b[n - 1] = -1.;
            b
        };

        let mut x = vec![0.0; n];

        let diag = vec![2.0; n];

        let sol = (0..n)
            .map(|i| if i % 2 == 0 { 1. } else { -1. })
            .collect::<Vec<f64>>();

        let start = std::time::Instant::now();

        let mut iter_count = 0;
        while x
            .iter()
            .zip(sol.iter())
            .map(|(a, b)| (a - b).abs())
            .reduce(f64::max)
            .unwrap()
            >= 1e-3
        {
            iter_count += 1;
            SparseMat::jacobi_iterate(&mut x, &b, &diag, &mat);
        }

        println!(
            "n = {n}, time used in {iter_count} jacobi iteration = {} ms",
            (std::time::Instant::now() - start).as_millis()
        );

        let full_mat = SparseMat::new(n, n, elems.chain((0..n).map(|i| (i, i, 2.0))));
        let y = full_mat.mul_vec(&x);
        println!(
            "backward error = {:e}",
            y.iter()
                .zip(b.iter())
                .map(|(m, n)| (m - n).abs())
                .reduce(f64::max)
                .unwrap()
        );

        assert!(x
            .iter()
            .zip(sol.iter())
            .all(|(a, b)| a.abs_diff_eq(b, 1e-3)));
    }

    #[test]
    fn test_guass_seidel_iterate_0() {
        let mat = SparseMat::full_from_vec(3, 3, vec![3., 2., -1., 1., 4., 2., -1., 1., 5.]);
        let mat = mat.transpose();
        let mut x = vec![0.0; 3];
        let b = [4., 1., 1.];

        (0..21).for_each(|i| {
            mat.gauss_seidel_iterate(&mut x, &b);
            if i < 2 {
                println!("x = {x:?}")
            }
        });

        (0..x.len()).for_each(|i| assert!(x[i].abs_diff_eq(&[2., -1., 1.][i], 1e-7)))
    }

    #[test]
    fn test_guass_seidel_iterate_1() {
        let n = 100;
        let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
        let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));
        let elems_2 = (0..n).map(|i| (i, i, 3.0));

        let elems = elems_0.chain(elems_1).chain(elems_2);
        let mat = SparseMat::new(n, n, elems);

        let b = {
            let mut b = vec![1.; n];
            b[0] = 2.;
            b[n - 1] = 2.;
            b
        };

        let mut x = vec![0.0; n];

        let start = std::time::Instant::now();

        let mut iter_count = 0;
        while x
            .iter()
            .map(|val| (val - 1.).abs())
            .reduce(f64::max)
            .unwrap()
            >= 1e-6
        {
            iter_count += 1;
            mat.gauss_seidel_iterate(&mut x, &b);
        }

        println!(
            "n = {n}, time used in {iter_count} gauss seidel iteration = {} ms",
            (std::time::Instant::now() - start).as_millis()
        );

        let y = mat.mul_vec(&x);
        println!(
            "backward error = {:e}",
            y.iter()
                .zip(b.iter())
                .map(|(m, n)| (m - n).abs())
                .reduce(f64::max)
                .unwrap()
        );

        assert!(x.iter().all(|y| y.abs_diff_eq(&1.0, 1e-6)));
    }

    #[test]
    fn test_guass_seidel_iterate_2() {
        let n = 100;
        let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, 1.0));
        let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, 1.0));
        let elems_2 = (0..n).map(|i| (i, i, 2.0));

        let elems = elems_0.chain(elems_1).chain(elems_2);
        let mat = SparseMat::new(n, n, elems.clone());

        let b = {
            let mut b = vec![0.; n];
            b[0] = 1.;
            b[n - 1] = -1.;
            b
        };

        let mut x = vec![0.0; n];

        let sol = (0..n)
            .map(|i| if i % 2 == 0 { 1. } else { -1. })
            .collect::<Vec<f64>>();

        let start = std::time::Instant::now();

        let mut iter_count = 0;
        while x
            .iter()
            .zip(sol.iter())
            .map(|(a, b)| (a - b).abs())
            .reduce(f64::max)
            .unwrap()
            >= 1e-3
        {
            iter_count += 1;
            mat.gauss_seidel_iterate(&mut x, &b);
        }

        println!(
            "n = {n}, time used in {iter_count} gauss seidel iteration = {} ms",
            (std::time::Instant::now() - start).as_millis()
        );

        let y = mat.mul_vec(&x);
        println!(
            "backward error = {:e}",
            y.iter()
                .zip(b.iter())
                .map(|(m, n)| (m - n).abs())
                .reduce(f64::max)
                .unwrap()
        );

        assert!(x
            .iter()
            .zip(sol.iter())
            .all(|(a, b)| a.abs_diff_eq(b, 1e-3)));
    }

    #[test]
    fn test_sor_guass_seidel_iterate_0() {
        let n = 100;
        let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, -1.0));
        let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, -1.0));
        let elems_2 = (0..n).map(|i| (i, i, 3.0));

        let elems = elems_0.chain(elems_1).chain(elems_2);
        let mat = SparseMat::new(n, n, elems);

        let b = {
            let mut b = vec![1.; n];
            b[0] = 2.;
            b[n - 1] = 2.;
            b
        };

        let mut x = vec![0.0; n];

        let start = std::time::Instant::now();

        let mut iter_count = 0;
        while x
            .iter()
            .map(|val| (val - 1.).abs())
            .reduce(f64::max)
            .unwrap()
            >= 1e-6
        {
            iter_count += 1;
            mat.sor_gauss_seidel_iterate(&mut x, &b, 1.2);
        }

        println!(
            "n = {n}, time used in {iter_count} sor gauss seidel iteration = {} ms",
            (std::time::Instant::now() - start).as_millis()
        );

        let y = mat.mul_vec(&x);
        println!(
            "backward error = {:e}",
            y.iter()
                .zip(b.iter())
                .map(|(m, n)| (m - n).abs())
                .reduce(f64::max)
                .unwrap()
        );

        assert!(x.iter().all(|y| y.abs_diff_eq(&1.0, 1e-6)));
    }

    #[test]
    fn test_sor_guass_seidel_iterate_1() {
        let n = 100;
        let elems_0 = (0..(n - 1)).map(|i| (i + 1, i, 1.0));
        let elems_1 = (0..(n - 1)).map(|i| (i, i + 1, 1.0));
        let elems_2 = (0..n).map(|i| (i, i, 2.0));

        let elems = elems_0.chain(elems_1).chain(elems_2);
        let mat = SparseMat::new(n, n, elems.clone());

        let b = {
            let mut b = vec![0.; n];
            b[0] = 1.;
            b[n - 1] = -1.;
            b
        };

        let mut x = vec![0.0; n];

        let sol = (0..n)
            .map(|i| if i % 2 == 0 { 1. } else { -1. })
            .collect::<Vec<f64>>();

        let start = std::time::Instant::now();

        let mut iter_count = 0;
        while x
            .iter()
            .zip(sol.iter())
            .map(|(a, b)| (a - b).abs())
            .reduce(f64::max)
            .unwrap()
            >= 1e-3
        {
            iter_count += 1;
            mat.sor_gauss_seidel_iterate(&mut x, &b, 1.5);
        }

        println!(
            "n = {n}, time used in {iter_count} sor gauss seidel iteration = {} ms",
            (std::time::Instant::now() - start).as_millis()
        );

        let y = mat.mul_vec(&x);
        println!(
            "backward error = {:e}",
            y.iter()
                .zip(b.iter())
                .map(|(m, n)| (m - n).abs())
                .reduce(f64::max)
                .unwrap()
        );

        assert!(x
            .iter()
            .zip(sol.iter())
            .all(|(a, b)| a.abs_diff_eq(b, 1e-3)));
    }
}
