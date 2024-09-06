pub trait MatShape {
    fn shape(&self) -> (usize, usize);

    fn is_square(&self) -> bool {
        self.shape().0 == self.shape().1
    }

    fn shape_eq(&self, other: &Self) -> bool {
        self.shape().0 == other.shape().0 && self.shape().1 == other.shape().1
    }
}
