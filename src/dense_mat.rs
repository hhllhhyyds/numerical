use std::{
    iter::Sum,
    ops::{Add, Index, IndexMut, Sub},
};

use approx::AbsDiffEq;

use crate::{
    mat_traits::{MatOps, MatShape},
    permutation::Permutation,
    FloatCore,
};

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMat<T> {
    nx: usize,
    ny: usize,
    data: Vec<T>,
}

impl<T: FloatCore + Sum> DenseMat<T> {
    pub fn mul_mat(&self, other: &Self) -> Self {
        assert!(self.nx == other.ny);

        let nx = other.nx;
        let ny = self.ny;

        let mut mat = Self::zeros(nx, ny);

        for iy in 0..ny {
            for ix in 0..nx {
                mat[(ix, iy)] = (0..self.nx).map(|j| self[(j, iy)] * other[(ix, j)]).sum();
            }
        }

        mat
    }

    pub fn mat_norm(&self) -> Option<T> {
        assert!(self.is_square());
        (0..self.ny)
            .map(|iy| (0..self.nx).map(|ix| self[(ix, iy)].abs()).sum())
            .reduce(T::max)
    }

    pub fn condition_number(&self) -> Option<T> {
        if let Some(norm_0) = self.mat_norm() {
            let inv = self.inverse();
            let norm_1 = inv.mat_norm().unwrap();
            Some(norm_0 * norm_1)
        } else {
            None
        }
    }

    pub fn is_strictly_diagonally_dominant(&self) -> bool {
        assert!(self.is_square());
        (0..self.ny).all(|iy| {
            (0..self.nx)
                .filter(|ix| *ix != iy)
                .map(|ix| self[(ix, iy)].abs())
                .sum::<T>()
                < self[(iy, iy)].abs()
        })
    }

    pub fn gauss_seidel_iterate(&self, x_k: &mut [T], b: &[T]) {
        assert!(self.is_square());
        let n = self.nx();
        assert!(x_k.len() == n);
        assert!(b.len() == n);

        for iy in 0..n {
            let val = (b[iy]
                - (0..n)
                    .filter(|ix| *ix != iy)
                    .map(|ix| self[(ix, iy)] * x_k[ix])
                    .sum())
                / self[(iy, iy)];
            x_k[iy] = val;
        }
    }
}

impl<T: FloatCore> DenseMat<T> {
    pub fn zeros(nx: usize, ny: usize) -> Self {
        Self::new(nx, ny, vec![T::zero(); nx * ny])
    }

    pub fn identity(n: usize) -> Self {
        let mut mat = Self::zeros(n, n);
        (0..n).for_each(|i| mat[(i, i)] = T::one());
        mat
    }

    pub fn from_diagonal(diag: &[T]) -> Self {
        let mut mat = Self::zeros(diag.len(), diag.len());
        (0..diag.len()).for_each(|i| mat[(i, i)] = diag[i]);
        mat
    }

    pub fn gaussian_elimination_solve(mut self, b: &[T]) -> Vec<T> {
        assert!(self.is_square());
        assert!(self.ny == b.len());

        if self.nx == 0 {
            return vec![];
        }

        let mut b = b.to_vec();
        let mut sol = vec![T::zero(); self.nx];

        for ix in 0..self.nx {
            for iy in (ix + 1)..self.ny {
                let coe = -self[(ix, iy)] / self[(ix, ix)];
                self[(ix, iy)] = T::zero();
                for jx in (ix + 1)..self.nx {
                    self[(jx, iy)] = self[(jx, iy)] + self[(jx, ix)] * coe;
                }
                b[iy] = b[iy] + b[ix] * coe;
            }
        }

        for iy in (0..self.ny).rev() {
            sol[iy] = b[iy];
            for ix in (iy + 1)..self.nx {
                sol[iy] = sol[iy] - self[(ix, iy)] * sol[ix];
            }
            sol[iy] = sol[iy] / self[(iy, iy)];
        }

        sol
    }

    pub fn lu(&self) -> (Self, Self) {
        assert!(self.is_square());

        let mut u_mat = self.clone();
        let mut l_mat = Self::new(self.nx, self.ny, vec![T::zero(); self.nx * self.ny]);

        for ix in 0..self.nx {
            l_mat[(ix, ix)] = T::one();
            for iy in (ix + 1)..self.ny {
                let coe = u_mat[(ix, iy)] / u_mat[(ix, ix)];
                u_mat[(ix, iy)] = T::zero();
                l_mat[(ix, iy)] = coe;
                for jx in (ix + 1)..self.nx {
                    u_mat[(jx, iy)] = u_mat[(jx, iy)] - u_mat[(jx, ix)] * coe;
                }
            }
        }

        (l_mat, u_mat)
    }

    pub fn back_substitute_lower_triangle(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());
        assert!(self.is_square());

        let mut y = vec![];
        for iy in 0..self.ny {
            let mut val = b[iy];
            for ix in 0..iy {
                val = val - self[(ix, iy)] * y[ix];
            }
            val = val / self[(iy, iy)];
            y.push(val);
        }

        y
    }

    pub fn back_substitute_upper_triangle(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());
        assert!(self.is_square());

        let mut y = vec![T::zero(); self.ny];

        for iy in (0..self.ny).rev() {
            y[iy] = b[iy];
            for ix in (iy + 1)..self.nx {
                y[iy] = y[iy] - self[(ix, iy)] * y[ix];
            }
            y[iy] = y[iy] / self[(iy, iy)];
        }

        y
    }

    pub fn lu_solve(&self, b: &[T]) -> Vec<T> {
        let (l, u) = self.lu();
        let c = l.back_substitute_lower_triangle(b);
        u.back_substitute_upper_triangle(&c)
    }

    pub fn plu(&self) -> (Permutation, Self, Self) {
        assert!(self.is_square());

        let mut u_mat = self.clone();
        let mut p = Permutation::new(self.nx);

        for ix in 0..u_mat.nx {
            let mut max_abs_iy = ix;
            let mut max_abs = u_mat[(ix, max_abs_iy)].abs();
            for iy in (ix + 1)..u_mat.ny {
                if u_mat[(ix, iy)].abs() > max_abs {
                    max_abs_iy = iy;
                    max_abs = u_mat[(ix, iy)].abs();
                }
            }
            if max_abs_iy != ix {
                u_mat.swap_row(ix, max_abs_iy);
                p.swap(ix, max_abs_iy);
            }

            for iy in (ix + 1)..u_mat.ny {
                let coe = u_mat[(ix, iy)] / u_mat[(ix, ix)];
                u_mat[(ix, iy)] = coe;
                for jx in (ix + 1)..self.nx {
                    u_mat[(jx, iy)] = u_mat[(jx, iy)] - u_mat[(jx, ix)] * coe;
                }
            }
        }

        let mut l_mat = Self::zeros(u_mat.nx, u_mat.ny);

        for ix in 0..u_mat.nx {
            for iy in ix..u_mat.ny {
                if ix == iy {
                    l_mat[(ix, iy)] = T::one();
                } else {
                    l_mat[(ix, iy)] = u_mat[(ix, iy)];
                    u_mat[(ix, iy)] = T::zero();
                }
            }
        }

        (p, l_mat, u_mat)
    }

    pub fn plu_solve(&self, b: &[T]) -> Vec<T> {
        let (p, l, u) = self.plu();
        Self::solve_plu(&p, &l, &u, b)
    }

    pub fn solve_plu(p: &Permutation, l: &Self, u: &Self, b: &[T]) -> Vec<T> {
        let pb = p.mul_vec(b);
        let c = l.back_substitute_lower_triangle(&pb);
        u.back_substitute_upper_triangle(&c)
    }

    pub fn inverse(&self) -> Self {
        let n = self.nx;
        let (p, l, u) = self.plu();
        let data = (0..n)
            .map(|i| {
                let mut v = vec![T::zero(); n];
                v[i] = T::one();
                Self::solve_plu(&p, &l, &u, &v)
            })
            .collect::<Vec<_>>()
            .concat();

        Self::new(n, n, data)
    }
}

impl<T: Clone> DenseMat<T> {
    pub fn transpose(&self) -> Self {
        let mut m = self.clone();
        m.nx = self.ny;
        m.ny = self.nx;
        (0..self.nx)
            .for_each(|ix| (0..self.ny).for_each(|iy| m[(iy, ix)] = self[(ix, iy)].clone()));
        m
    }

    pub fn diagonal(&self) -> Vec<T> {
        assert!(self.is_square());
        (0..self.nx).map(|i| self[(i, i)].clone()).collect()
    }
}

impl<T> DenseMat<T> {
    pub fn new(nx: usize, ny: usize, data: Vec<T>) -> Self {
        assert!(data.len() == nx * ny);
        Self { nx, ny, data }
    }

    pub fn swap_row(&mut self, iy_0: usize, iy_1: usize) {
        (0..self.nx).for_each(|ix| self.swap_element((ix, iy_0), (ix, iy_1)))
    }

    pub fn swap_element(&mut self, pos_a: (usize, usize), pos_b: (usize, usize)) {
        let pos_a = self.index_2d_to_1d(pos_a);
        let pos_b = self.index_2d_to_1d(pos_b);
        self.data.swap(pos_a, pos_b)
    }

    #[inline]
    fn index_2d_to_1d(&self, idx: (usize, usize)) -> usize {
        idx.0 * self.ny + idx.1
    }
}

impl<T> MatShape for DenseMat<T> {
    fn shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }
}

impl<T: FloatCore + Sum> MatOps<T> for DenseMat<T> {
    fn mul_vec(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());

        (0..self.ny)
            .map(|iy| (0..self.nx).map(|ix| self[(ix, iy)] * b[ix]).sum())
            .collect::<Vec<T>>()
    }
}

impl<T> Index<(usize, usize)> for DenseMat<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[self.index_2d_to_1d(index)]
    }
}

impl<T> IndexMut<(usize, usize)> for DenseMat<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.index_2d_to_1d(index);
        &mut self.data[idx]
    }
}

impl<T: FloatCore> Add for &DenseMat<T> {
    type Output = DenseMat<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.shape_eq(rhs));

        let mut m = DenseMat::zeros(self.nx, self.ny);
        (0..self.nx).for_each(|ix| {
            (0..self.ny).for_each(|iy| m[(ix, iy)] = self[(ix, iy)] + rhs[(ix, iy)])
        });

        m
    }
}

impl<T: FloatCore> Sub for &DenseMat<T> {
    type Output = DenseMat<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.shape_eq(rhs));

        let mut m = DenseMat::zeros(self.nx, self.ny);
        (0..self.nx).for_each(|ix| {
            (0..self.ny).for_each(|iy| m[(ix, iy)] = self[(ix, iy)] - rhs[(ix, iy)])
        });

        m
    }
}

impl<T: FloatCore + AbsDiffEq<Epsilon = T>> AbsDiffEq for DenseMat<T> {
    type Epsilon = T;

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.nx == other.nx
            && self.ny == other.ny
            && self
                .data
                .iter()
                .zip(other.data.iter())
                .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }

    fn default_epsilon() -> Self::Epsilon {
        <T as AbsDiffEq>::default_epsilon()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::{assert_abs_diff_eq, AbsDiffEq};

    fn hilbert_mat<T: FloatCore>(n: usize) -> DenseMat<T> {
        let data = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| T::one() / T::from(i + j + 1).unwrap())
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
            .concat();
        DenseMat::new(n, n, data)
    }

    #[test]
    fn test_gaussian_elimination_solve_0() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., 2., 1., 1., -1., -2., 1.]);
        let b = vec![3., 3., -6.];
        let sol = mat.gaussian_elimination_solve(&b);
        assert_abs_diff_eq!(sol[0], 3.0);
        assert_abs_diff_eq!(sol[1], 1.0);
        assert_abs_diff_eq!(sol[2], 2.0);
    }

    #[test]
    fn test_gaussian_elimination_solve_1() {
        let mat = DenseMat::new(1, 1, vec![2.]);
        let b = vec![3.];
        let sol = mat.gaussian_elimination_solve(&b);
        assert_abs_diff_eq!(sol[0], 1.5);
        assert!(sol.len() == 1);
    }

    #[test]
    fn test_gaussian_elimination_solve_2() {
        let mat = DenseMat::new(0, 0, Vec::<f64>::new());
        let b = vec![];
        let sol = mat.gaussian_elimination_solve(&b);
        assert!(sol.is_empty());
    }

    #[test]
    fn test_gaussian_elimination_solve_3() {
        let n = 6;
        let mat = hilbert_mat::<f64>(n);
        let x = vec![1.0; n];
        let b = mat.mul_vec(&x);
        let sol = mat.gaussian_elimination_solve(&b);
        sol.iter().for_each(|x| assert!(x.abs_diff_eq(&1.0, 1e-9)));

        let n = 8;
        let mat = hilbert_mat::<f64>(n);
        let x = vec![1.0; n];
        let b = mat.mul_vec(&x);
        let sol = mat.gaussian_elimination_solve(&b);
        sol.iter().for_each(|x| assert!(x.abs_diff_eq(&1.0, 1e-6)));
    }

    #[test]
    fn test_gaussian_elimination_solve_4() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., -1., -2., 3., -1., -2., 1.]);
        let b = vec![3., 3., -6.];
        let sol = mat.gaussian_elimination_solve(&b);
        assert!(sol.iter().filter(|x| x.is_nan()).count() > 0);
        println!("sol = {sol:?}")
    }

    #[test]
    fn test_lu_0() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., 2., 1., 1., -1., -2., 1.]);
        let (l_mat, u_mat) = mat.lu();
        assert!(l_mat.mul_mat(&u_mat).abs_diff_eq(&mat, 1e-14))
    }

    #[test]
    fn test_lu_1() {
        let mat = DenseMat::new(3, 3, vec![3., 6., 3., 1., 3., 1., 2., 4., 5.]);
        let (l_mat, u_mat) = mat.lu();
        assert!(l_mat.mul_mat(&u_mat).abs_diff_eq(&mat, 1e-14))
    }

    #[test]
    fn test_lu_2() {
        let n = 8;
        let mat = hilbert_mat::<f64>(n);
        let (l_mat, u_mat) = mat.lu();
        assert!(l_mat.mul_mat(&u_mat).abs_diff_eq(&mat, 1e-14))
    }

    #[test]
    fn test_lu_3() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., 2., 1., 1., -1., -2., 1.]);
        let b = vec![3., 3., -6.];
        let sol_0 = mat.lu_solve(&b);
        let sol_1 = mat.gaussian_elimination_solve(&b);
        sol_0
            .iter()
            .zip(sol_1.iter())
            .for_each(|(a, b)| assert!(a == b));
    }

    #[test]
    fn test_lu_4() {
        let n = 8;
        let mat = hilbert_mat::<f64>(n);
        let b = vec![1.0; n];
        let sol_0 = mat.lu_solve(&b);
        let sol_1 = mat.gaussian_elimination_solve(&b);
        sol_0
            .iter()
            .zip(sol_1.iter())
            .for_each(|(a, b)| assert!(a == b));
        println!("sol = {sol_0:?}")
    }

    #[test]
    fn test_lu_5() {
        let mat = DenseMat::new(3, 3, vec![1., 2., -3., -1., -2., 3., -1., -2., 1.]);
        let (l_mat, u_mat) = mat.lu();
        assert!(l_mat.data.iter().filter(|x| x.is_nan()).count() > 0);
        assert!(u_mat.data.iter().filter(|x| x.is_nan()).count() > 0);
    }

    #[test]
    fn test_mul_mat() {
        let m0 = DenseMat::new(1, 4, vec![1.0; 4]);
        let m1 = DenseMat::new(4, 1, vec![1.0; 4]);

        let m3 = DenseMat::new(1, 1, vec![4.0; 1]);
        let m4 = DenseMat::new(4, 4, vec![1.0; 16]);

        assert!(m0.mul_mat(&m1).abs_diff_eq(&m4, 1e-14));
        assert!(m1.mul_mat(&m0).abs_diff_eq(&m3, 1e-14));
    }

    #[test]
    fn test_plu_0() {
        let mat = DenseMat::new(3, 3, vec![2., 4., 1., 1., 4., 3., 5., -4., 1.]);
        let (p, l_mat, u_mat) = mat.plu();
        println!("p = {p:?}");
        println!("l_mat = {l_mat:?}");
        println!("u_mat = {u_mat:?}");
    }

    #[test]
    fn test_plu_1() {
        let mat = DenseMat::new(3, 3, vec![2., 4., 1., 1., 4., 3., 5., -4., 1.]);
        let b = [5., 0., 6.];
        let sol_0 = mat.lu_solve(&b);
        let sol_1 = mat.plu_solve(&b);

        sol_0
            .iter()
            .zip(sol_1.iter())
            .for_each(|(a, b)| assert!(a == b));
    }

    #[test]
    fn test_plu_2() {
        let mat = DenseMat::new(2, 2, vec![1e-20, 1., 1., 2.]);
        let b = [1., 4.];
        let sol_1 = mat.plu_solve(&b);
        let sol_0 = mat.gaussian_elimination_solve(&b);

        sol_0
            .iter()
            .zip(&[0., 1.])
            .for_each(|(a, b)| assert!(a.abs_diff_eq(b, 1e-10)));

        sol_1
            .iter()
            .zip(&[2., 1.])
            .for_each(|(a, b)| assert!(a.abs_diff_eq(b, 1e-10)));
    }

    #[test]
    fn test_mat_inverse_0() {
        let mat = DenseMat::new(3, 3, vec![2., 4., 1., 1., 4., 3., 5., -4., 1.]);
        let mat_inv = mat.inverse();
        let mat_mul_inv = mat.mul_mat(&mat_inv);
        assert!(mat_mul_inv.abs_diff_eq(&DenseMat::identity(3), 1e-14))
    }

    #[test]
    fn test_mat_inverse_1() {
        let mat = DenseMat::new(2, 2, vec![1., 1.0001, 1., 1.]);
        let cond = mat.condition_number().unwrap();
        assert!(cond.abs_diff_eq(&40004., 1e-3))
    }

    #[test]
    fn test_mat_inverse_2() {
        let mat = hilbert_mat::<f64>(6);
        let cond = mat.condition_number().unwrap();
        println!("condition number of hilbert 6 = {cond:e}");
        assert!(cond.abs_diff_eq(&2.9070279e7, 1.0));

        let mat = hilbert_mat::<f64>(10);
        let cond = mat.condition_number().unwrap();
        println!("condition number of hilbert 10 = {cond:e}");
    }

    #[test]
    fn test_strictly_diagonally_dominant() {
        let mat_0 = DenseMat::new(3, 3, vec![3., 2., 1., 1., -5., 6., -1., 2., 8.]);
        assert!(mat_0.is_strictly_diagonally_dominant());

        let mut mat_1 = DenseMat::new(3, 3, vec![3., 1., 9., 2., 8., 2., 6., 1., -2.]);
        assert!(!mat_1.is_strictly_diagonally_dominant());

        mat_1.swap_row(0, 2);
        assert!(mat_1.is_strictly_diagonally_dominant());
    }

    #[test]
    fn test_transpose() {
        let mat_0 = DenseMat::new(2, 3, vec![3., 2., 1., 1., -5., 6.]);
        let mat_1 = DenseMat::new(3, 2, vec![3., 1., 2., -5., 1., 6.]);
        let mat_0_transpose = mat_0.transpose();

        println!("mat_1 = {mat_1:?}, mat_0 transpose = {:?}", mat_0_transpose);
        assert!(mat_0_transpose.abs_diff_eq(&mat_1, f64::EPSILON))
    }

    #[test]
    fn test_mat_add_sub() {
        let mat_0 = DenseMat::new(2, 3, vec![3., 2., 1., 1., -5., 6.]);
        let mat_1 = DenseMat::new(2, 3, vec![6., 4., 2., 2., -10., 12.]);
        let mat_2 = &mat_0 + &mat_0;
        let mat_3 = &mat_0 - &mat_0;
        assert!(mat_2.abs_diff_eq(&mat_1, f64::EPSILON));
        assert!(mat_3.abs_diff_eq(&DenseMat::zeros(2, 3), f64::EPSILON));
    }

    #[test]
    fn test_jacobi_iterate_0() {
        let mat = DenseMat::new(2, 2, vec![3., 1., 1., 2.]);
        let diagonal = mat.diagonal();
        let l_plus_u = &mat - &(DenseMat::from_diagonal(&diagonal));
        let mut x = vec![0.0; 2];
        let b = [5.0; 2];

        (0..20).for_each(|_| {
            DenseMat::jacobi_iterate(&mut x, &b, &diagonal, &l_plus_u);
        });

        (0..x.len()).for_each(|i| assert!(x[i].abs_diff_eq(&[1.0, 2.0][i], 1e-7)))
    }

    #[test]
    fn test_guass_seidel_iterate_0() {
        let mat = DenseMat::new(3, 3, vec![3., 2., -1., 1., 4., 2., -1., 1., 5.]);

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
}
