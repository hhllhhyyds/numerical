use std::iter::Sum;

use crate::{
    matrix::{mat_shape::MatShape, permutation::Permutation},
    FloatCore,
};

use super::DenseMat;

impl<T: FloatCore> DenseMat<T> {
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
        let mut l_mat = Self::from_vec(self.nx, self.ny, vec![T::zero(); self.nx * self.ny]);

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
                u_mat.swap_iy(ix, max_abs_iy);
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

        Self::from_vec(n, n, data)
    }
}

impl<T: FloatCore + Sum> DenseMat<T> {
    pub fn gauss_seidel_iterate(&self, x_k: &mut [T], b: &[T]) {
        let n = {
            assert!(self.is_square());
            self.nx()
        };
        assert!(x_k.len() == n);
        assert!(b.len() == n);

        (0..n).for_each(|iy| {
            let val = (b[iy]
                - (0..n)
                    .filter(|ix| *ix != iy)
                    .map(|ix| self[(ix, iy)] * x_k[ix])
                    .sum())
                / self[(iy, iy)];
            x_k[iy] = val;
        })
    }
}

impl<T: crate::Float + FloatCore + Sum> DenseMat<T> {
    pub fn cholesky_factorization(&mut self) -> Option<Self> {
        assert!(self.is_square());

        let mut r_mat = DenseMat::zeros(self.nx, self.ny);

        for iy in 0..self.ny {
            if self[(iy, iy)] < T::zero() {
                return None;
            }
            r_mat[(iy, iy)] = self[(iy, iy)].sqrt();
            for ix in (iy + 1)..self.nx {
                r_mat[(ix, iy)] = self[(ix, iy)] / r_mat[(iy, iy)];
            }
            for a in (iy + 1)..self.nx {
                for b in (iy + 1)..self.ny {
                    self[(a, b)] = self[(a, b)] - r_mat[(a, iy)] * r_mat[(b, iy)];
                }
            }
        }

        Some(r_mat)
    }
}
