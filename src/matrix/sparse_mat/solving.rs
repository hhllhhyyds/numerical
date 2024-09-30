use crate::{matrix::mat_shape::MatShape, FloatCore};

use super::SparseMat;

impl<T: FloatCore> SparseMat<T> {
    pub fn back_substitute_lower_triangle(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());
        assert!(self.is_square());

        let mut y = vec![];

        let transpose = self.transpose();

        let mut iter = transpose.data.iter();
        let mut elem = iter.next();
        #[allow(clippy::needless_range_loop)]
        for ix_now in 0..transpose.nx {
            let mut y_i = b[ix_now];
            while let Some((&(ix, iy), &val)) = elem {
                if ix == ix_now {
                    if iy < ix_now {
                        y_i = y_i - y[iy] * val;
                    }
                } else {
                    break;
                }
                elem = iter.next();
            }
            y.push(y_i / *transpose.get((ix_now, ix_now)).unwrap_or(&T::zero()));
        }

        y
    }

    pub fn back_substitute_upper_triangle(&self, b: &[T]) -> Vec<T> {
        assert!(self.nx == b.len());
        assert!(self.is_square());

        let mut y = vec![T::zero(); self.nx];

        let transpose = self.transpose();

        let mut iter = transpose.data.iter().rev();
        let mut elem = iter.next();
        for ix_now in (0..transpose.nx).rev() {
            y[ix_now] = b[ix_now];
            while let Some((&(ix, iy), &val)) = elem {
                if ix == ix_now {
                    if iy > ix_now {
                        y[ix_now] = y[ix_now] - y[iy] * val;
                    }
                } else {
                    break;
                }
                elem = iter.next();
            }
            y[ix_now] = y[ix_now] / *transpose.get((ix_now, ix_now)).unwrap_or(&T::zero());
        }

        y
    }

    pub fn gauss_seidel_iterate(&self, x_k: &mut [T], b: &[T]) {
        assert!(self.is_square());
        let n = self.nx();
        assert!(x_k.len() == n);
        assert!(b.len() == n);

        let transpose = self.transpose();

        let mut iter = transpose.data.iter();
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
            x_k[ix_now] =
                (b[ix_now] - sum) / *transpose.get((ix_now, ix_now)).unwrap_or(&T::zero());
        }
    }

    pub fn sor_gauss_seidel_iterate(&self, x_k: &mut [T], b: &[T], omega: T) {
        assert!(self.is_square());
        let n = self.nx();
        assert!(x_k.len() == n);
        assert!(b.len() == n);

        let transpose = self.transpose();

        let mut iter = transpose.data.iter();
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
                + omega * (b[ix_now] - sum)
                    / *transpose.get((ix_now, ix_now)).unwrap_or(&T::zero());
        }
    }
}
