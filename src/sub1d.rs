const FUNCTION_HANDLE1: fn(&Vec<f32>) -> f32 = |x|  0.01 * (x[0] + 50.4).powi(4) - 0.7 * (x[0] + 50.4).powi(2) + 0.1 * x[0];
const FUNCTION_HANDLE2: fn(&Vec<f32>) -> f32 = |x|  0.5 * x[0].powi(2) + x[0].powi(2) * x[0].sin().powi(2) + x[0];
const FUNCTION_HANDLE3: fn(&Vec<f32>) -> f32 = |x|  x[0].abs();
const FUNCTION_HANDLE4: fn(&Vec<f32>) -> f32 = |x|  (x[0]-50.0).powi(2);
pub fn func1(x :&Vec<f32> ) -> f32 {
    FUNCTION_HANDLE1(x)
}pub fn func2(x :&Vec<f32> ) -> f32 {
    FUNCTION_HANDLE2(x)
}pub fn func3(x :&Vec<f32> ) -> f32 {
    FUNCTION_HANDLE3(x)
}pub fn func4(x :&Vec<f32> ) -> f32 {
    FUNCTION_HANDLE4(x)
}