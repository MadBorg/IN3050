
# Standard import and functions
# Run this cell first
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -np.power(x, 4) + 2 * np.power(x, 3) + 2 * np.power(x, 2) - x

def df(x):
    return -4 * np.power(x, 3) + 6 * np.power(x, 2) + 4 * x - 1
    # return -4 * x**3 + 6 * x**2 + 4 * x - 1


# Add your solution here
# x = np.linspace(-2, 3, int(1e4))
# plt.plot(x,f(x), label="f(x)" )
# plt.plot(x,df(x), label="df(x)")
# plt.legend()
# plt.show()

# Add your solution here
gamma = 0.1
iterations = int( 1e3 )
x = np.zeros(iterations)
x[0] = 1
for k in range(iterations-1):
    x[k+1] = x[k] + gamma * df(x[k])

Y = f(np.linspace(-2,3,iterations))
Exhaustive = max(Y)
print(f"gradient ascent: f({x[-1]}) = {f(x[-1])}, Exhaustive : f : {np.max(Y)}")

import IPython as IP; IP.embed()