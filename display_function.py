import re
import inspect
import subprocess
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def function_handle(x):
    fx = 0.01 * (x + 5.4)**4 - 0.7 * (x + 5.4)**2 + 0.1 * x
    return fx

def sin_square_function(x):
    fx = 0.5 * x**2 + x**2 * np.sin(x)**2 + x
    return fx


def function_to_rust_string(function_handle):
    text = inspect.getsource(function_handle)
    lines = text.split("\n")
    variables = re.findall(r"(?<=\().*(?=\))", lines[0])
    function_str = lines[1:-1]
    function_str = function_str[0].split("=")[1]
    function_str = re.sub(r'np\.sin\(' + variables[0] + r'\)',  variables[0] + r'.sin()', function_str)
    function_str = re.sub(r'(?<=\*\*)\d',  r'.powi(\g<0>)', function_str)
    function_str = re.sub(r'\*\*',  '', function_str)
    return function_str

# print(function_to_rust_string(sin_square_function))

def write_sub_rs(rust_str1, rust_str2):
    with open(r'src\sub.rs', 'w') as f:
        f.write("const FUNCTION_HANDLE1: fn(f32) -> f32 = |x| {};\n".format(rust_str1))
        f.write("const FUNCTION_HANDLE2: fn(f32) -> f32 = |x| {};\n".format(rust_str2))
        f.write("pub fn func1(x: f32 ) -> f32 {\n    FUNCTION_HANDLE1(x)\n}")
        f.write("pub fn func2(x: f32 ) -> f32 {\n    FUNCTION_HANDLE2(x)\n}")

rust_str1 = function_to_rust_string(function_handle)
rust_str2 = function_to_rust_string(sin_square_function)
write_sub_rs(rust_str1, rust_str2)



p = subprocess.Popen('cargo run --release', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
for line in iter(p.stdout.readline, ''):
    print(line[0:-1])
p.stdout.close()
retval = p.wait()
print("ran simulation")

def show_progress (filepath, calc_function):
    df = pd.read_csv(filepath, sep=",")
    print(df)
    ax = plt.subplot()

    t = np.arange(min(df["XVal"]), max(df["XVal"]), (max(df["XVal"])-min(df["XVal"]))/100)
    s = calc_function(t)
    line, = plt.plot(t, s, lw=2)

    # plt.plot(df["XVal"][0], function_handle(df["XVal"][0]), 'bo')

    x = df['XVal'].values
    y = [calc_function(val) for val in df['XVal'].values]
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 
    ax.plot(x,y, marker="o")
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    lendf= len(df) -1

    plt.plot([df["XVal"][lendf]-df["x-minus"][lendf], df["XVal"][lendf]+df["x-plus"][lendf]], [calc_function(df["XVal"][lendf])]*2, lw=2, linestyle="solid", color="red")
    # plt.ylim(-5, 5)
    plt.show()

show_progress(r"data\powerfour.txt", function_handle)
show_progress(r"data\quadratic_sinus.txt", sin_square_function)