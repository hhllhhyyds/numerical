pub trait MatShape {
    fn nx(&self) -> usize;
    fn ny(&self) -> usize;

    fn is_square(&self) -> bool {
        self.nx() == self.ny()
    }

    fn shape_eq(&self, other: &Self) -> bool {
        self.nx() == other.nx() && self.ny() == other.ny()
    }
}
