#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    pub fn new(n: usize) -> Self {
        Permutation((0..n).collect())
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j)
    }

    pub fn mul_vec<T: Clone>(&self, v: &[T]) -> Vec<T> {
        assert!(self.0.len() == v.len());

        let mut permu_v = v.to_vec();

        for i in 0..v.len() {
            permu_v[i] = v[self.0[i]].clone();
        }

        permu_v
    }
}
