const FUNCTION_HANDLE: fn(f32) -> f32 = |x|  0.5 * x.powi(4) - 0.7 * x.powi(2) + 0.1 * x;
pub fn func1(x: f32 ) -> f32 {
    FUNCTION_HANDLE(x)
}