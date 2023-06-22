use std::f64::consts::PI;
const FUNCTION_HANDLE1: fn(&Vec<f64>) -> f64 = |x|  0.01 * (x[0] + 50.4).powi(4) - 0.7 * (x[0] + 50.4).powi(2) + 0.1 * x[0] + x[1].powi(2);
const FUNCTION_HANDLE2: fn(&Vec<f64>) -> f64 = |x|  0.5 * x[0].powi(2) + x[0].powi(2) * x[0].sin().powi(2) + x[0]+ x[1].powi(2);
const FUNCTION_HANDLE3: fn(&Vec<f64>) -> f64 = |x|  x[0].abs()+ x[1].abs();
const FUNCTION_HANDLE4: fn(&Vec<f64>) -> f64 = |x|  (x[0]-50.0).powi(2)+ (x[1]-20.0).powi(2);
const FUNCTION_HANDLE_ROSENBROCK: fn(&Vec<f64>) -> f64 = |x|  100.0*(x[1] - x[0].powi(2)).powi(2) + (1.0 - (x[0])).powi(2);
const FUNCTION_HANDLE_RASTRIGIN: fn(&Vec<f64>) -> f64 = |x|  20.0 + (x[0]).powi(2) - 10.0*(2.0*PI*x[0]).cos() + (x[1]).powi(2) - 10.0*(2.0*PI*x[1]).cos();
pub fn func1(x: &Vec<f64> ) -> f64 {
    FUNCTION_HANDLE1(x)
}pub fn func2(x: &Vec<f64> ) -> f64 {
    FUNCTION_HANDLE2(x)
}pub fn func3(x: &Vec<f64> ) -> f64 {
    FUNCTION_HANDLE3(x)
}pub fn func4(x: &Vec<f64> ) -> f64 {
    FUNCTION_HANDLE4(x)
}pub fn func5(x: &Vec<f64> ) -> f64 {
    FUNCTION_HANDLE_ROSENBROCK(x)
}pub fn func6(x: &Vec<f64> ) -> f64 {
    FUNCTION_HANDLE_RASTRIGIN(x)
}