const FUNCTION_HANDLE1: fn(f32) -> f32 = |x|  0.01 * (x + 50.4).powi(4) - 0.7 * (x + 50.4).powi(2) + 0.1 * x;
const FUNCTION_HANDLE2: fn(f32) -> f32 = |x|  0.5 * x.powi(2) + x.powi(2) * x.sin().powi(2) + x;
const FUNCTION_HANDLE3: fn(f32) -> f32 = |x|  x.abs();
const FUNCTION_HANDLE4: fn(f32) -> f32 = |x|  (x-50.0).powi(2);
pub fn func1(x: f32 ) -> f32 {
    FUNCTION_HANDLE1(x)
}pub fn func2(x: f32 ) -> f32 {
    FUNCTION_HANDLE2(x)
}pub fn func3(x: f32 ) -> f32 {
    FUNCTION_HANDLE3(x)
}pub fn func4(x: f32 ) -> f32 {
    FUNCTION_HANDLE4(x)
}