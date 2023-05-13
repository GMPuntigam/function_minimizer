from matplotlib import pyplot as plt
from random import seed
from random import random
seed(2)

x_values = [random() for _ in range(5)]
x_values.sort()
mean = (sum(x_values)/len(x_values))
y_values = [-(_-2)**2 for _ in range(5)]
mean_y = (sum(y_values)/len(y_values))
dx = [val - mean for val in x_values]
# dy = [val - mean_y for val in y_values]
momentum = sum([dx[i]*y_values[i] for i in range(len(x_values))])
print(momentum)
ax = plt.subplot()
ax.plot(x_values, y_values, marker="o")
ax.plot(mean, mean_y, marker="o", color="r")
X, Y, U, V = zip([mean, mean_y, momentum/2, 0.1])
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=['red', 'green', 'yellow'])
plt.show()