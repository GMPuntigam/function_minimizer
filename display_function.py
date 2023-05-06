import re
import inspect
import subprocess
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
def function_handle(x):
    fx = 0.5 * x**4 - 0.7 * x**2 + 0.1 * x
    return fx



def function_to_rust_string(function_handle):
    lines = inspect.getsource(function_handle)
    function_str = lines.split("\n")[1:-2]
    function_str = function_str[0].split("=")[1]
    function_str = re.sub(r'(?<=\*\*)\d',  r'.powi(\g<0>)', function_str)
    function_str = re.sub(r'\*\*',  '', function_str)
    return function_str



def write_sub_rs(rust_str):
    with open(r'src\sub.rs', 'w') as f:
        f.write("const FUNCTION_HANDLE: fn(f64) -> f64 = |x| {};\n".format(rust_str))
        f.write("pub fn func1(x: f64 ) -> f64 {\n    FUNCTION_HANDLE(x)\n}")

rust_str = function_to_rust_string(function_handle)
write_sub_rs(rust_str)



p = subprocess.Popen('cargo run --release', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
for line in iter(p.stdout.readline, ''):
    print(line[0:-1])
p.stdout.close()
retval = p.wait()
print("ran simulation")

df = pd.read_csv(r"data\out.txt", sep=",")
print(df)
ax = plt.subplot()

t = np.arange(-5.0, 5.0, 0.01)
s = function_handle(t)
line, = plt.plot(t, s, lw=2)

plt.plot(df["XVal"][0], function_handle(df["XVal"][0]), 'bo')
# ax.arrow(df["XVals"][0], 1, df["Delta"][0], 0, head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.ylim(-5, 5)
plt.show()