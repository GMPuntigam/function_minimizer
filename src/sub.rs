const FUNCTION_HANDLE: fn(f64) -> f64 = |x|  0.4 * x.powi(4) - 0.7 * x.powi(2) + 0.1 * x;
pub fn func1(x: f64 ) -> f64 {
    FUNCTION_HANDLE(x)
}