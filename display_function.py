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

def abs_function(x):
    fx = np.abs(x)
    return fx

def x_squared(x):
    fx = (x-5.0)**2
    return fx


def function_to_rust_string(function_handle):
    text = inspect.getsource(function_handle)
    lines = text.split("\n")
    variables = re.findall(r"(?<=\().*(?=\))", lines[0])
    function_str = lines[1:-1]
    function_str = function_str[0].split("=")[1]
    function_str = re.sub(r'np\.sin\(' + variables[0] + r'\)',  variables[0] + r'.sin()', function_str)
    function_str = re.sub(r'np\.abs\(' + variables[0] + r'\)',  variables[0] + r'.abs()', function_str)
    function_str = re.sub(r'(?<=\*\*)\d',  r'.powi(\g<0>)', function_str)
    function_str = re.sub(r'\*\*',  '', function_str)
    return function_str

# print(function_to_rust_string(sin_square_function))

def write_sub_rs(function_strs):
    with open(r'src\sub.rs', 'w') as f:
        for i, rust_str in enumerate(function_strs):
            f.write("const FUNCTION_HANDLE{}: fn(f32) -> f32 = |x| {};\n".format(i+1, rust_str))
        for i, rust_str in enumerate(function_strs):
            f.write("pub fn func{}(x: f32 ) -> f32 ".format(i+1) +"{" + "\n    FUNCTION_HANDLE{}(x)\n".format(i+1) + "}")

rust_str1 = function_to_rust_string(function_handle)
rust_str2 = function_to_rust_string(sin_square_function)
rust_str3 = function_to_rust_string(abs_function)
rust_str4 = function_to_rust_string(x_squared)
write_sub_rs([rust_str1, rust_str2, rust_str3, rust_str4])



# p = subprocess.Popen('cargo run --release', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
# for line in iter(p.stdout.readline, ''):
#     print(line[0:-1])
# p.stdout.close()
# retval = p.wait()
# print("ran simulation")

def show_step(function, step, df):
    df_view = df[df.index==step]
    samplepoints = int((len(df.columns) - 3)/2)
    # print(df_view)
    
    x_vals = []
    fx_vals = []
    for point in range(samplepoints):
        x_vals.append(df_view["x{}".format(point)][step])
        fx_vals.append(df_view["f(x{})".format(point)][step])
    ax = plt.subplot()
    t = np.arange(min(x_vals), max(x_vals), (max(x_vals) - min(x_vals))/1000)
    s = function(t)
    ax.plot(x_vals,fx_vals, marker="o")
    ax.plot(df_view["XVal"][step],function(df_view["XVal"][step]), marker="o", color="red")
    line, = plt.plot(t, s, lw=2)
    plt.plot([df_view["XVal"][step]-df_view["x-minus"][step], df_view["XVal"][step]+df_view["x-plus"][step]], [function(df_view["XVal"][step])]*2, lw=2, linestyle="solid", color="red")
    plt.show()


def show_progress (filepath, function, closeup =False):
    df = pd.read_csv(filepath, sep=",")
    if closeup:
        for step in range(len(df)):
            show_step(function, step, df)
    # print(df)
    ax = plt.subplot()
    if closeup:
        t = np.arange(-5, 5, 0.01)
    else:
        t = np.arange(min(df["XVal"]), max(df["XVal"]), (max(df["XVal"])-min(df["XVal"]))/100)
    s = function(t)
    line, = plt.plot(t, s, lw=2)
    
    # plt.plot(df["XVal"][0], function_handle(df["XVal"][0]), 'bo')

    x = df['XVal'].values
    y = [function(val) for val in df['XVal'].values]
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 
    ax.plot(x,y, marker="o")
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    lendf= len(df) -1

    plt.plot([df["XVal"][lendf]-df["x-minus"][lendf], df["XVal"][lendf]+df["x-plus"][lendf]], [function(df["XVal"][lendf])]*2, lw=2, linestyle="solid", color="red")
    if closeup:
        plt.ylim(-10, 10)
        plt.xlim(-5, 5)
    plt.show()

show_progress(r"data\powerfour.txt", function_handle)
show_progress(r"data\powerfour.txt", function_handle, True)
show_progress(r"data\quadratic_sinus.txt", sin_square_function)
show_progress(r"data\quadratic_sinus.txt", sin_square_function, True)
show_progress(r"data\abs.txt", abs_function)
show_progress(r"data\abs.txt", abs_function, True)
show_progress(r"data\x-squared.txt", x_squared)
show_progress(r"data\x-squared.txt", x_squared, True)