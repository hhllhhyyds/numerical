use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseMat<T> {
    nx: usize,
    ny: usize,
    data: BTreeMap<(usize, usize), T>,
}

pub mod constructor;
pub mod index;
pub mod ops;
pub mod solving;
