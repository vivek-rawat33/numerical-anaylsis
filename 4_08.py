import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, pi

def f(t):
    return np.exp(-t**2)

def trapezoidal(x, n=1000):
    a, b = 0, x
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += 2 * f(a + i*h)
    return (h/2) * s

def simpson(x, n=1000):
    if n % 2 != 0: n += 1
    a, b = 0, x
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += 4*f(a + i*h) if i % 2 != 0 else 2*f(a + i*h)
    return (h/3) * s

def gauss_legendre(x):
    xi = np.array([-1/sqrt(3), 1/sqrt(3)])
    w = np.array([1, 1])
    a, b = 0, x
    return (b - a)/2 * np.sum(w * np.exp(-((b - a)/2 * xi + (b + a)/2)**2))

X = np.linspace(0, 5, 30)
erf_trap = [2/sqrt(pi) * trapezoidal(x) for x in X]
erf_simp = [2/sqrt(pi) * simpson(x) for x in X]
erf_gauss = [2/sqrt(pi) * gauss_legendre(x) for x in X]

from math import erf
erf_true = [erf(x) for x in X]

# Plot comparison
plt.figure(figsize=(9,6))
plt.plot(X, erf_true, 'k-', label='True erf(x)')
plt.plot(X, erf_trap, '*', label='Trapezoidal')
plt.plot(X, erf_simp, 'g-', label='Simpson')
plt.plot(X, erf_gauss, 'b--', label='Gauss–Legendre')
plt.xlabel('x')
plt.ylabel('erf(x)')
plt.title('Comparison of Numerical Methods for erf(x)')
plt.legend()
plt.grid(True)
plt.show()
