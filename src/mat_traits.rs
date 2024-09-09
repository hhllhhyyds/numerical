pub trait MatShape {
    fn shape(&self) -> (usize, usize);

    fn nx(&self) -> usize {
        self.shape().0
    }

    fn ny(&self) -> usize {
        self.shape().1
    }

    fn is_square(&self) -> bool {
        self.shape().0 == self.shape().1
    }

    fn shape_eq(&self, other: &Self) -> bool {
        self.shape().0 == other.shape().0 && self.shape().1 == other.shape().1
    }
}

pub trait MatOps<T: crate::FloatCore>: MatShape {
    fn mul_vec(&self, b: &[T]) -> Vec<T>;

    fn jacobi_iterate(x_k: &mut [T], b: &[T], diagonal: &[T], l_plus_u: &Self) {
        assert!(l_plus_u.is_square());
        let n = l_plus_u.nx();
        assert!(x_k.len() == n);
        assert!(b.len() == n);
        assert!(diagonal.len() == n);

        let x = l_plus_u.mul_vec(x_k);
        (0..n).for_each(|i| x_k[i] = (b[i] - x[i]) / diagonal[i]);
    }
}
