import numpy as np
import matplotlib.pyplot as plt
import math
# True function
f = lambda x: np.exp(x)
x = np.linspace(-1, 1, 400)
y_true = f(x)

# ----- Maclaurin (Taylor) series -----
def maclaurin_exp(x, n=6):
    return sum((x**k) /math.factorial(k) for k in range(n))

# ----- Chebyshev polynomials (manual) -----
def T(n, x):
    if n == 0: return np.ones_like(x)
    if n == 1: return x
    T0, T1 = np.ones_like(x), x
    for k in range(2, n+1):
        T0, T1 = T1, 2*x*T1 - T0
    return T1

# Simple Chebyshev approximation using discrete cosine projection
def chebyshev_exp(x, n=6):
    N = 200  # number of sample points
    xs = np.cos(np.pi*(np.arange(N)+0.5)/N)  # Chebyshev nodes
    ys = f(xs)
    coeffs = []
    for k in range(n):
        Tk = T(k, xs)
        ck = (2/N) * np.sum(ys * Tk)
        coeffs.append(ck)
    coeffs[0] /= 2
    # reconstruct
    y = np.zeros_like(x)
    for k, c in enumerate(coeffs):
        y += c * T(k, x)
    return y

# Evaluate
y_mac = maclaurin_exp(x, 6)
y_cheb = chebyshev_exp(x, 6)

# Plot
plt.plot(x, y_true, 'k', label='True $e^x$')
plt.plot(x, y_mac, 'r--', label='Maclaurin (6 terms)')
plt.plot(x, y_cheb, 'b-.', label='Chebyshev (6 terms)')
plt.legend()
plt.title("Chebyshev vs. Maclaurin Approximation of $e^x$")
plt.grid(True)
plt.show()
