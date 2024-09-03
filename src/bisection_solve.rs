use std::ops::Range;

use num_traits::float::FloatCore;

pub fn bisection_solve<T: FloatCore>(
    f: &dyn Fn(T) -> T,
    x_tol: T,
    y_tol: T,
    search_range: Range<T>,
) -> Range<T> {
    assert!(x_tol > T::zero());
    assert!(y_tol > T::zero());

    let two = T::one() + T::one();

    let (mut a, mut b) = (search_range.start, search_range.end);
    let mut fa = f(a);

    loop {
        if (b - a) / two < x_tol {
            break;
        }

        let c = (a + b) / two;
        let fc = f(c);
        if fc.abs() < y_tol {
            break;
        }

        if fa * fc < T::zero() {
            b = c;
        } else {
            (a, fa) = (c, fc);
        }
    }

    a..b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::Polynomial;
    use approx::AbsDiffEq;

    #[test]
    fn test_bisection_solve_0() {
        let solution_range = bisection_solve(
            &|x| Polynomial::new(&[1.0, -3.0, 2.0]).nest_multiply(x),
            1e-4,
            1e-12,
            1.5..2.9,
        );
        let solution = (solution_range.start + solution_range.end) / 2.0;
        assert!(solution.abs_diff_eq(&2.0, 1e-4), "solution = {solution}");
    }

    #[test]
    fn test_bisection_solve_1() {
        let solution_range = bisection_solve(
            &|x| Polynomial::new(&[1.0_f32, -3.0, 2.0]).nest_multiply(x),
            1e-6,
            1e-4,
            1.5..2.9,
        );
        let solution = (solution_range.start + solution_range.end) / 2.0;
        assert!(solution.abs_diff_eq(&2.0, 1e-4), "solution = {solution}");
    }

    #[test]
    fn test_bisection_solve_2() {
        let solution_range = bisection_solve(
            &|x| Polynomial::new(&[1.0_f32, -3.0, 2.0]).nest_multiply(x),
            1e-6,
            1e-8,
            1.5..2.9,
        );
        let solution = (solution_range.start + solution_range.end) / 2.0;
        assert!(solution.abs_diff_eq(&2.0, 1e-6), "solution = {solution}");
    }
}
