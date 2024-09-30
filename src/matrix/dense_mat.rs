#[derive(Debug, Clone, PartialEq, Default)]
pub struct DenseMat<T> {
    nx: usize,
    ny: usize,
    data: Vec<T>,
}

mod constructor;
mod index;
mod ops;
mod properties;
mod solving;
