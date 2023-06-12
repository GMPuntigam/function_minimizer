import os
import pandas as pd
import matplotlib.pyplot as plt
dir = r"example_runs\2023-6-12"
dir2 = r"P:\Data\code\Rust-minimizer\pso_python"
path_rosenbrock_pso = r"P:\Data\code\Rust-minimizer\pso_python\rosenbrock_approach.txt"

def visualise_approach(filepath, filename):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    

    df = pd.read_csv(filepath, sep=";")
    ax.plot(df['n_eval'], df['f_min'] + abs(min(df['f_min'])), color='blue', lw=2)

    filecompare = os.sep.join([dir2,filename])
    if os.path.exists(filecompare):
        df2 = pd.read_csv(filecompare, sep=";")
        ax.plot(df2['n_eval'], df2['f_min'] + abs(min(df2['f_min'])), color='red', lw=2)

    ax.set_yscale('log')
    plt.show()

for filename in os.listdir(dir):
    if filename.endswith("_approach.txt"): 
        visualise_approach(os.sep.join([dir,filename]), filename)

