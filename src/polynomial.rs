use crate::FloatCore;

pub struct Polynomial<T: FloatCore> {
    coes: Vec<T>,
    base_points: Option<Vec<T>>,
}

impl<T: FloatCore> Polynomial<T> {
    pub fn new(coes: &[T]) -> Self {
        assert!(!coes.is_empty(), "coes can not be empty");
        Self {
            coes: coes.to_vec(),
            base_points: None,
        }
    }

    pub fn with_base_points(mut self, bps: &[T]) -> Self {
        assert!(
            bps.len() + 1 == self.coes.len(),
            "bps.len() + 1 != coes.len()"
        );
        self.base_points = Some(bps.to_vec());
        self
    }

    pub fn nest_multiply(&self, x: T) -> T {
        let n = self.coes.len();
        let mut y = self.coes[0];

        if let Some(bps) = &self.base_points {
            #[allow(clippy::needless_range_loop)]
            for i in 0..(n - 1) {
                y = y * (x - bps[i]) + self.coes[i + 1];
            }
        } else {
            for i in 0..(n - 1) {
                y = y * x + self.coes[i + 1];
            }
        }

        y
    }
}

#[test]
fn test_nest_multiply_0() {
    let poly = Polynomial::new(&[-0.5, 0.5, 0.5, 1.0]).with_base_points(&[3.0, 2.0, 0.0]);
    let y = poly.nest_multiply(1.0);
    assert!(y == 0.0, "y = {y}");
}

#[test]
fn test_nest_multiply_1() {
    let poly = Polynomial::new(&[2.0, 3.0, -3.0, 5.0, -1.0]);

    let y = poly.nest_multiply(-2.0);
    assert!(y == -15.0, "y = {y}");

    let y = poly.nest_multiply(-1.0);
    assert!(y == -10.0, "y = {y}");

    let y = poly.nest_multiply(0.0);
    assert!(y == -1.0, "y = {y}");

    let y = poly.nest_multiply(1.0);
    assert!(y == 6.0, "y = {y}");

    let y = poly.nest_multiply(2.0);
    assert!(y == 53.0, "y = {y}");
}

#[test]
fn test_nest_multiply_2() {
    let poly = Polynomial::new(&[1.0; 51]);

    let x = 1.00001;

    let y_0 = poly.nest_multiply(x);
    println!("y_0 = {y_0}");

    let y_1 = (x.powi(51) - 1.0) / (x - 1.0);
    println!("y_1 = {y_1}");

    let abs_relative_diff = (y_0 - y_1).abs() / y_1.abs();
    println!("abs_relative_diff = {abs_relative_diff:e}");

    assert!(abs_relative_diff < 2e-12);
}

#[test]
fn test_nest_multiply_3() {
    let poly = Polynomial::new(
        &(0..100)
            .into_iter()
            .map(|i| (-1.0).powi(i % 2 + 1))
            .collect::<Vec<_>>(),
    );

    let x = 1.00001;

    let y_0 = poly.nest_multiply(x);
    println!("y_0 = {y_0}");

    let y_1 = (1.0 - (x).powi(100)) / (x + 1.0);
    println!("y_1 = {y_1}");

    let abs_relative_diff = (y_0 - y_1).abs() / y_1.abs();
    println!("abs_relative_diff = {abs_relative_diff:e}");

    assert!(abs_relative_diff < 2e-12);
}
