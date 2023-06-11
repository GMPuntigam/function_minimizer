import os
import pandas as pd
import matplotlib.pyplot as plt
dir = r"example_runs\2023-6-11"

def visualise_approach(filepath):
    df = pd.read_csv(filepath, sep=";")
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)

    line, = ax.plot(df['n_eval'], df['f_min'] + abs(min(df['f_min'])), color='blue', lw=2)

    ax.set_yscale('log')
    plt.show()

for filename in os.listdir(dir):
    if filename.endswith("_approach.txt"): 
        visualise_approach(os.sep.join([dir,filename]))

