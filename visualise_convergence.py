import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
dir = r"P:\Data\code\Rust-minimizer\function_minimizer\example_runs\2023-6-22"
dir2 = r"P:\Data\code\Rust-minimizer\pso_python"
dir3 = r"P:\Data\code\Rust-minimizer\scikit-opt"

dict_fn_to_title = {
    "powerfour": "Test Function 1",
    "quadratic-sinus": "Test Function 2",
    "x-abs+sin": "Test Function 3",
    "rosenbrock": "Rosenbrock",
    "rastrigin": "Rastrigin",
    "x-squared": "Offset Sphere",
    "abs": "Absolute Value",
    "beale": "Beale",
    "ackley": "Ackley",
    "bukin": "Bukin N.6",
}

# dict_fn_to_min = {
#     "powerfour": "Test Function 1",
#     "quadratic-sinus": "Test Function 2",
#     "x-abs+sin": "Test Function 3",
#     "rosenbrock": "Rosenbrock",
#     "rastrigin": "Rastrigin",
#     "x-squared": "Offset Sphere",
#     "abs": "Absolute Value",
# }


def visualise_approach(filepath, filename):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    

    df = pd.read_csv(filepath, sep=";")
    filecompare = os.sep.join([dir2,filename])
    df2 = pd.read_csv(filecompare, sep=";")
    filecompare = os.sep.join([dir3,filename])
    df3 = pd.read_csv(filecompare, sep=";")
    overall_best_min = min(min(df['f_min']), min(df2['f_min']), min(df3['f_min']))

    line1 = ax.plot(df['n_eval'], (df['f_min'] - overall_best_min), color='tab:blue', lw=2)
    line2 = ax.plot(df2['n_eval'], (df2['f_min'] - overall_best_min), color='tab:orange', lw=2)
    line3 = ax.plot(df3['n_eval'], (df3['f_min'] - overall_best_min), color='tab:green', lw=2)
    ax.set_title(dict_fn_to_title[filename.split("_approach")[0]])
    # ax.set_yscale('logit')
    ax.set_yscale('log')
    # ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
    ax.set_xlabel('Number of function evaluations')
    ax.set_ylabel('Objective function evaluation')
    ax.legend((line1[0], line2[0], line3[0]), ('ENN', 'PSO', 'DE'))
    ax.yaxis.grid(True)
    # ax.set_ylim(0.1,0.999999999)
    plt.show()

for filename in os.listdir(dir):
    if filename.endswith("_approach.txt"): 
        visualise_approach(os.sep.join([dir,filename]), filename)

