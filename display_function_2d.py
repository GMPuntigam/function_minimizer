import re
import inspect
import subprocess
import pandas as pd
from matplotlib import pyplot as plt, ticker, colors as colors
import numpy as np
from math import floor, ceil
import os

MAXPLOTS_X = 4
MAXPLOTS_Y = 5

def function_handle(x, y):
    fx = 0.01 * (x + 50.4)**4 - 0.7 * (x + 50.4)**2 + 0.1 * x +y**2
    return fx

def sin_square_function(x, y):
    fx = 0.5 * x**2 + x**2 * np.sin(x)**2 + x + y**2
    return fx

def abs_function(x, y):
    fx = np.abs(x) + np.abs(y)
    return fx

def x_squared(x, y):
    fx = (x-50.0)**2 + (y-20.0)**2
    return fx

def x_abs_sin(x, y):
    fx = (x-3.14)**2 + (y-2.72)**2+ np.sin(3.0*x +1.41) + np.sin(4.0*y -1.73)
    return fx

def rastrigin(x,y):
    fx = 20.0 + x**2 - 10.0*np.cos((2.0*np.pi*x)) + y**2 - 10.0*np.cos((2.0*np.pi*y))
    return fx

def rosenbrock(x,y):
    fx = 100.0*(y - x**2)**2 + (1.0 - x)**2
    return fx

# def function_to_rust_string(function_handle):
#     text = inspect.getsource(function_handle)
#     lines = text.split("\n")
#     variables = re.findall(r"(?<=\().*(?=\))", lines[0])
#     function_str = lines[1:-1]
#     function_str = function_str[0].split("=")[1]
#     function_str = re.sub(r'np\.sin\(' + variables[0] + r'\)',  variables[0] + r'.sin()', function_str)
#     function_str = re.sub(r'np\.abs\(' + variables[0] + r'\)',  variables[0] + r'.abs()', function_str)
#     function_str = re.sub(r'(?<=\*\*)\d',  r'.powi(\g<0>)', function_str)
#     function_str = re.sub(r'\*\*',  '', function_str)
#     return function_str

# # print(function_to_rust_string(sin_square_function))

# def write_sub_rs(function_strs):
#     with open(r'src\sub.rs', 'w') as f:
#         for i, rust_str in enumerate(function_strs):
#             f.write("const FUNCTION_HANDLE{}: fn(f32) -> f32 = |x| {};\n".format(i+1, rust_str))
#         for i, rust_str in enumerate(function_strs):
#             f.write("pub fn func{}(x: f32 ) -> f32 ".format(i+1) +"{" + "\n    FUNCTION_HANDLE{}(x)\n".format(i+1) + "}")

# rust_str1 = function_to_rust_string(function_handle)
# rust_str2 = function_to_rust_string(sin_square_function)
# rust_str3 = function_to_rust_string(abs_function)
# rust_str4 = function_to_rust_string(x_squared)
# write_sub_rs([rust_str1, rust_str2, rust_str3, rust_str4])



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

def show_progress_surf (filepath, function, closeup =False, steps = False):
    from matplotlib import cm
    df = pd.read_csv(filepath, sep=";")
    lendf= len(df) -1
    # n_plots = MAXPLOTS_X*MAXPLOTS_Y
    # if steps:
    #     slides = ceil((lendf+1)/n_plots)
    #     for slide in range(slides):
    #         fig, axs = plt.subplots(MAXPLOTS_Y, MAXPLOTS_X)
    #         for step in range(n_plots):
    #             show_step(axs, function, step, df, slide)
    #         plt.show()
    # print(df)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # if closeup:
    #     t1 = np.arange(min(df["XVal"]), df["XVal"][lendf]-10, ((df["XVal"][lendf]-10)-min(df["XVal"]))/500)
    #     t2 = np.arange(df["XVal"][lendf]-10, df["XVal"][lendf]+10, 0.01)
    #     t3 = np.arange(df["XVal"][lendf]+10, max(df["XVal"]) + 10, (max(df["XVal"]) + 10-(df["XVal"][lendf]+10))/500)
    #     t= np.concatenate((t1, t2, t3), axis=None)
    # else:
    #     t = np.arange(min(df["XVal"]), max(df["XVal"]) + 5, (max(df["XVal"])-min(df["XVal"]) + 5)/5000)
    # s = function(t)
    # line, = plt.plot(t, s, lw=2)
    
    # plt.plot(df["XVal"][0], function_handle(df["XVal"][0]), 'bo')
    x1 = []
    x2 = []
    for x in df['XVal'].values:
        x_parts = x.split(",")
        x1.append(float(x_parts[0]))
        x2.append(float(x_parts[1]))
    xz = [function(x1[i], x2[i]) for i in range(len(df['XVal'].values))]

    if closeup:
        x = np.linspace(x1[-1]-10,x1[-1]+ 10, 500)
        y = np.linspace(x2[-1]-10,x2[-1]+ 10, 500)
    else:
        x = np.linspace(-100, 100, 500)
        y = np.linspace(-50, 50, 500)
    z = np.array([function(i, j) for j in y for i in x])
    X, Y = np.meshgrid(x, y)
    Z = z.reshape(500, 500)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

    if steps:
        u = np.diff(x1)
        v = np.diff(x2)
        w = np.diff(xz)
        pos_x1 = x1[:-1] + u/2
        pos_x2 = x2[:-1] + v/2
        pos_z = xz[:-1] + w/2
        norm = np.sqrt(u**2+v**2+w**2) 
        ax.plot(x1,x2,xz, marker="o")
        ax.quiver(pos_x1, pos_x2, pos_z, u/norm, v/norm, w/norm, pivot="middle")
        lendf= len(df) -1
    ax.plot(x1[-1],x2[-1], xz[-1], marker="x", color="red")
    df_last = df[df["Step"] == len(df)]
    x_minus_parts = df_last["x-minus"].values[0].split(",")
    x_plus_parts = df_last["x-plus"].values[0].split(",")
    plt.plot([x1[-1]-float(x_minus_parts[0]), x1[-1]+float(x_plus_parts[0])], [x2[-1]]*2, [xz[-1]]*2, lw=2, linestyle="solid", color="red")
    plt.plot([x1[-1]]*2, [x2[-1]-float(x_minus_parts[1]), x2[-1]+float(x_plus_parts[1])], [xz[-1]]*2, lw=2, linestyle="solid", color="red")
    if closeup:
        plt.ylim(x2[-1]-10, x2[-1]+10)
        plt.xlim(x1[-1]-10,x1[-1]+10)
        ax.set_zlim(xz[-1]-0.1,xz[-1]+1)
    plt.show()



def show_progress (filepath, function, closeup =False, steps = False, logcontours = False):
    df = pd.read_csv(filepath, sep=";")
    lendf= len(df) -1
    ax = plt.subplot()
    x1 = []
    x2 = []
    for x in df['XVal'].values:
        x_parts = x.split(",")
        x1.append(float(x_parts[0]))
        x2.append(float(x_parts[1]))
    xz = [function(x1[i], x2[i]) for i in range(len(df['XVal'].values))]

    if closeup and function != rosenbrock:
        lims = [x1[-1]-5, x1[-1]+ 5, x2[-1]-5, x2[-1]+ 5]
    elif closeup and function == rosenbrock:
        lims = [x1[-1]-3, x1[-1]+ 3, x2[-1]-3, x2[-1]+ 3]    
    else:
        lims = [-100, 100, -50, 50]
    x = np.linspace(lims[0], lims[1], 100)
    y = np.linspace(lims[2], lims[3], 100)
    z = np.array([function(i, j) for j in y for i in x])
    X, Y = np.meshgrid(x, y)
    Z = z.reshape(100, 100)
    if logcontours:
        img = ax.imshow(Z, norm=colors.LogNorm(vmin=1 ,vmax= abs(Z.min()) + Z.max()), extent=lims, origin='lower', cmap='viridis', alpha=0.5)
    elif function == rosenbrock and closeup:
        img = ax.imshow(Z, norm=colors.LogNorm(vmin=1 , vmax= abs(Z.min()) + Z.max()), extent=lims, origin='lower', cmap='viridis', alpha=0.5)
    else:
        img = ax.imshow(Z, extent=lims, origin='lower', cmap='viridis', alpha=0.5)
    if logcontours:
        contours = ax.contour(X, Y, Z, 10,locator=ticker.LogLocator() , colors='black', alpha=0.4, zorder=6)
    elif function == rosenbrock and closeup:
        contours = ax.contour(X, Y, Z, 30,locator=ticker.LogLocator() , colors='black', alpha=0.4, zorder=6)
    else:
        contours = ax.contour(X, Y, Z, 10, colors='black', alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    plt.colorbar(img)
    if steps:
        u = np.diff(x1)
        v = np.diff(x2)
        pos_x1 = x1[:-1] + u/2
        pos_x2 = x2[:-1] + v/2
        norm = np.sqrt(u**2+v**2) 
        ax.plot(x1,x2, marker="o")
        ax.quiver(pos_x1, pos_x2, u/norm, v/norm, angles="xy", zorder=5, pivot="mid", scale=100)
        lendf= len(df) -1
    ax.plot(x1[-1],x2[-1], marker="x", color="red")
    df_last = df[df["Step"] == len(df)]
    x_minus_parts = df_last["x-minus"].values[0].split(",")
    x_plus_parts = df_last["x-plus"].values[0].split(",")
    ax.plot([x1[-1]-float(x_minus_parts[0]), x1[-1]+float(x_plus_parts[0])], [x2[-1]]*2, lw=2, linestyle="solid", color="red")
    ax.plot([x1[-1]]*2, [x2[-1]-float(x_minus_parts[1]), x2[-1]+float(x_plus_parts[1])], lw=2, linestyle="solid", color="red")
    if function == rastrigin:
        ax.plot(0,0, marker="x", color="lightgreen")
    if function == rosenbrock:
        ax.plot(1,1, marker="x", color="lightgreen")
    if closeup:
        plt.ylim(x2[-1]-5, x2[-1]+5)
        plt.xlim(x1[-1]-5,x1[-1]+5)
    plt.show()

dir = r"example_runs\2023-5-20"

function_handle_dict = {"powerfour.txt": function_handle,
                        "abs.txt": abs_function,
                        "quadratic-sinus.txt": sin_square_function,
                        "x-abs+sin.txt": x_abs_sin,
                        "x-squared.txt": x_squared,
                        "rastringin.txt": rastrigin,
                        "rosenbrock.txt": rosenbrock}

for filename in os.listdir(dir):
    if filename.endswith(".txt"): 
        if filename == "abs.txt":
            logcontours = False
        else:  
            logcontours = True
        if filename in function_handle_dict.keys():
            show_progress(os.sep.join([dir,filename]), function_handle_dict[filename], steps = True, logcontours = logcontours)
            show_progress(os.sep.join([dir,filename]), function_handle_dict[filename], steps = False, closeup=True, logcontours = False)
    else:
        continue