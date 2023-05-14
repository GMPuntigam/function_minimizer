import re
import inspect
import subprocess
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from math import floor, ceil
import os

MAXPLOTS_X = 4
MAXPLOTS_Y = 5

def function_handle(x):
    fx = 0.01 * (x + 50.4)**4 - 0.7 * (x + 50.4)**2 + 0.1 * x
    return fx

def sin_square_function(x):
    fx = 0.5 * x**2 + x**2 * np.sin(x)**2 + x
    return fx

def abs_function(x):
    fx = np.abs(x)
    return fx

def x_squared(x):
    fx = (x-50.0)**2
    return fx

def x_abs_sin(x):
    fx = np.abs(x-3) + 1.5*np.sin(x)
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

def show_step(axs, function, step, df, slide):
    step_total = step + slide*MAXPLOTS_X*MAXPLOTS_Y
    df_view = df[df.index==(step_total)]
    samplepoints = int((len(df.columns) - 5)/2)
    # print(df_view)
    
    x_vals = []
    fx_vals = []
    if step_total not in df.index:
        return None
    for point in range(samplepoints):
        x_vals.append(df_view["x{}".format(point)][step_total])
        fx_vals.append(df_view["f(x{})".format(point)][step_total])
    ax = axs[floor(step/MAXPLOTS_X), step%MAXPLOTS_X]
    ax.set_title("Step {}".format(step_total))
    t = np.arange(min(x_vals), max(x_vals), (max(x_vals) - min(x_vals))/1000)
    s = function(t)
    ax.plot(x_vals,fx_vals, marker="o")
    ax.plot(df_view["XVal"][step_total],function(df_view["XVal"][step_total]), marker="o", color="red")
    line, = ax.plot(t, s, lw=2)
    ax.plot([df_view["XVal"][step_total]-df_view["x-minus"][step_total], df_view["XVal"][step_total]+df_view["x-plus"][step_total]], [function(df_view["XVal"][step_total])]*2, lw=2, linestyle="solid", color="red")
    


def show_progress (filepath, function, closeup =False, steps = False):
    df = pd.read_csv(filepath, sep=",")
    lendf= len(df) -1
    n_plots = MAXPLOTS_X*MAXPLOTS_Y
    if steps:
        slides = ceil((lendf+1)/n_plots)
        for slide in range(slides):
            fig, axs = plt.subplots(MAXPLOTS_Y, MAXPLOTS_X)
            for step in range(n_plots):
                show_step(axs, function, step, df, slide)
            plt.show()
    # print(df)
    ax = plt.subplot()
    if closeup:
        t1 = np.arange(min(df["XVal"]), df["XVal"][lendf]-10, ((df["XVal"][lendf]-10)-min(df["XVal"]))/500)
        t2 = np.arange(df["XVal"][lendf]-10, df["XVal"][lendf]+10, 0.01)
        t3 = np.arange(df["XVal"][lendf]+10, max(df["XVal"]) + 10, (max(df["XVal"]) + 10-(df["XVal"][lendf]+10))/500)
        t= np.concatenate((t1, t2, t3), axis=None)
    else:
        t = np.arange(min(df["XVal"]), max(df["XVal"]) + 5, (max(df["XVal"])-min(df["XVal"]) + 5)/5000)
    s = function(t)
    line, = plt.plot(t, s, lw=2)
    
    # plt.plot(df["XVal"][0], function_handle(df["XVal"][0]), 'bo')
    
    x = df['XVal'].values
    y = [function(val) for val in df['XVal'].values]
    if steps:
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u/2
        pos_y = y[:-1] + v/2
        norm = np.sqrt(u**2+v**2) 
        ax.plot(x,y, marker="o")
        ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
        lendf= len(df) -1
    ax.plot(x[-1],y[-1], marker="x", color="red")
    plt.plot([df["XVal"][lendf]-df["x-minus"][lendf], df["XVal"][lendf]+df["x-plus"][lendf]], [function(df["XVal"][lendf])]*2, lw=2, linestyle="solid", color="red")
    if closeup:
        plt.ylim(function(df["XVal"][lendf])-10, function(df["XVal"][lendf])+10)
        plt.xlim(min(df["XVal"][lendf]-5,-5)-5,max(df["XVal"][lendf]+5,5)+ 5)
    plt.show()

dir = "example_runs/2023-5-14"

function_handle_dict = {"powerfour.txt": function_handle,
                        "abs.txt": abs_function,
                        "quadratic-sinus.txt": sin_square_function,
                        "x-abs+sin.txt": x_abs_sin,
                        "x-squared.txt": x_squared}

for filename in os.listdir(dir):
    if filename.endswith(".txt"): 
        show_progress(os.sep.join([dir,filename]), function_handle_dict[filename])
        show_progress(os.sep.join([dir,filename]), function_handle_dict[filename], True)
         # print(os.path.join(directory, filename))
        continue
    else:
        continue