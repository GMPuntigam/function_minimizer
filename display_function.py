import re
import inspect

def function_handle(x):
    fx = 0.4 * x^4 - 0.7 * x^2 + 0.1 * x
    return fx



def function_to_rust_string(function_handle):
    lines = inspect.getsource(function_handle)
    function_str = lines.split("\n")[1:-2]
    function_str = function_str[0].split("=")[1]
    function_str = re.sub(r'(?<=\^)\d',  r'.powi(\g<0>)', function_str)
    function_str = re.sub(r'\^',  '', function_str)
    return function_str



def write_sub_rs(rust_str):
    with open(r'src\sub.rs', 'w') as f:
        f.write("const FUNCTION_HANDLE: fn(f64) -> f64 = |x| {};\n".format(rust_str))
        f.write("pub fn func1(x: f64 ) -> f64 {\n    FUNCTION_HANDLE(x)\n}")

rust_str = function_to_rust_string(function_handle)
write_sub_rs(rust_str)