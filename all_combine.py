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


# Gauss-Seidel Method until steady (3 decimal places)

def gauss_seidel(a, b):
    n = len(a)
    x = [0.0] * n   # initial guess inside function
    iteration = 0
    while True:
        iteration += 1
        x_old = x.copy()
        for i in range(n):
            s = sum(a[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / a[i][i]

        # stop if steady (difference < 0.001 → 3 decimals)
        if all(abs(x[i] - x_old[i]) < 0.001 for i in range(n)):
            break
    return [round(val, 3) for val in x], iteration


# Example
n = int(input("Enter the size of matrix (n): "))
A = []
for i in range(n):
    row = []
    print(f"Enter elements of row {i+1}:")
    for j in range(n):
        val = float(input(f"Element[{i+1},{j+1}]: "))
        row.append(val)
    A.append(row)

B = []
for i in range(n):
    val = float(input(f"Element for B[{i+1}] : "))
    B.append(val)

solution, iteration = gauss_seidel(A, B)
print(f"\nThe solution is {solution}, and number of iterations = {iteration}")

# Gauss–Jordan Method 
def gauss_jordan(A, b):
    n = len(b)
    aug = []
    for i in range(n):
        row = list(A[i]) + [b[i]]
        aug.append(row)
    for i in range(n):
        pivot = aug[i][i]
        for k in range(i, n+1):   
            aug[i][k] /= pivot
        for j in range(n):
            if j != i:
                factor = aug[j][i]
                for k in range(i, n+1):
                    aug[j][k] -= factor * aug[i][k]
    x = [aug[i][-1] for i in range(n)]
    return x
n = int(input("Enter the size of matrix (n): "))
matrix = []
for i in range(n):
    row = []
    print(f"Enter elements of row {i+1}:")
    for j in range(n):
        val = float(input(f"Element[{i+1},{j+1}]: "))
        row.append(val)
    matrix.append(row)
B=[]
for i in range(n):
    val= float(input(f"Element for B[{i+1}] :" ))
    B.append(val)


solution = gauss_jordan(matrix, B)
print("Solution:", solution)



# Gauss-Jordan method to find inverse of a matrix

def gauss_jordan_inverse(matrix):
    n = len(matrix)
    aug = [row + [0]*n for row in matrix]
    for i in range(n):
        aug[i][n+i] = 1
    
    for i in range(n):
        diag = aug[i][i]
        if diag  == 0:
            raise ValueError("Matrix is singular, cannot find inverse.")
        for j in range(2*n):
            aug[i][j] /= diag
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2*n):
                    aug[k][j] -= factor * aug[i][j]
    inverse = [row[n:] for row in aug]
    return inverse
A = [
    [2, 1],
    [5, 3]
]
invA = gauss_jordan_inverse(A)
print("Inverse of A:")
for row in invA:
    print(row)


import numpy as np
import matplotlib.pyplot as plt

# Differential equation
def f(x, y):
    return -y + x

# Exact solution
def y_exact(x):
    return 2*np.exp(-x) + x - 1

# Euler Method
def euler_method(f, x0, y0, h, n):
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x0 + i*h, y[i])
    return y

# Runge-Kutta 2nd Order
def rk2_method(f, x0, y0, h, n):
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        k1 = f(x0 + i*h, y[i])
        k2 = f(x0 + i*h + h/2, y[i] + h/2 * k1)
        y[i+1] = y[i] + h * k2
    return y

# Runge-Kutta 4th Order
def rk4_method(f, x0, y0, h, n):
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(n):
        xi = x0 + i*h
        k1 = f(xi, y[i])
        k2 = f(xi + h/2, y[i] + h/2*k1)
        k3 = f(xi + h/2, y[i] + h/2*k2)
        k4 = f(xi + h, y[i] + h*k3)
        y[i+1] = y[i] + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return y

# Initial conditions
x0 = 0
y0 = 1
X = 2

# Step sizes
step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

errors_euler, errors_rk2, errors_rk4 = [], [], []

# Error computation
for h in step_sizes:
    n = int((X - x0) / h)
    x = [x0 + i*h for i in range(n+1)]
    y_ex = [y_exact(xi) for xi in x]

    y_e = euler_method(f, x0, y0, h, n)
    y_r2 = rk2_method(f, x0, y0, h, n)
    y_r4 = rk4_method(f, x0, y0, h, n)

    errors_euler.append(max(abs(ye - ye_ex) for ye, ye_ex in zip(y_e, y_ex)))
    errors_rk2.append(max(abs(yr2 - ye_ex) for yr2, ye_ex in zip(y_r2, y_ex)))
    errors_rk4.append(max(abs(yr4 - ye_ex) for yr4, ye_ex in zip(y_r4, y_ex)))

# Print table
print(f"{'h':>10} {'Euler Error':>15} {'RK2 Error':>15} {'RK4 Error':>15}")
for i, h in enumerate(step_sizes):
    print(f"{h:10.6f} {errors_euler[i]:15.6e} {errors_rk2[i]:15.6e} {errors_rk4[i]:15.6e}")

# Plotting
plt.figure(figsize=(8,5))
plt.loglog(step_sizes, errors_euler, 'o-', label='Euler')
plt.loglog(step_sizes, errors_rk2, 's-', label='RK2')
plt.loglog(step_sizes, errors_rk4, '^-', label='RK4')

plt.xlabel('Step size h')
plt.ylabel('Maximum absolute error')
plt.title('Error vs Step Size')
plt.gca().invert_xaxis()
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()




#air drag physics problem
# Air drag physics problem (RK4 for coupled ODE system)

def rk4_system(f, g, x0, v0, T, h):
    n = int(T / h)
    t = [0]
    x = [x0]
    v = [v0]

    for i in range(n):
        ti = t[-1]
        xi = x[-1]
        vi = v[-1]

        # k-values for velocity ODE dv/dt = f(t, v)
        kv1 = f(ti, vi)
        kx1 = g(ti, xi, vi)

        kv2 = f(ti + h/2, vi + h/2*kv1)
        kx2 = g(ti + h/2, xi + h/2*kx1, vi + h/2*kv1)

        kv3 = f(ti + h/2, vi + h/2*kv2)
        kx3 = g(ti + h/2, xi + h/2*kx2, vi + h/2*kv2)

        kv4 = f(ti + h, vi + h*kv3)
        kx4 = g(ti + h, xi + h*kx3, vi + h*kv3)

        # Update values
        v_new = vi + h/6*(kv1 + 2*kv2 + 2*kv3 + kv4)
        x_new = xi + h/6*(kx1 + 2*kx2 + 2*kx3 + kx4)

        # Append
        t.append(ti + h)
        v.append(v_new)
        x.append(x_new)

    return t, x, v


# Example usage for free fall with linear drag
m = 70
g_val = 9.8
k = 12
v0 = 0
x0 = 0
T = 10
h = 0.1

# Acceleration function dv/dt
def f(t, v):
    return g_val - (k/m)*v

# dx/dt = v
def g(t, x, v):
    return v

# Solve ODE system
t, x, v = rk4_system(f, g, x0, v0, T, h)

# Print every 1 second (h = 0.1 → 10 steps per second)
for i in range(0, len(t), 10):
    print(f"t={t[i]:.1f} s, x={x[i]:.2f} m, v={v[i]:.2f} m/s")


import numpy as np
def LU(A):
    n = len(A)
    L = [[1 if i== j else 0 for j in range(n)] for i in range(n)]
    U = [[0.0]*n for i in range(n)]
    for k in range(n):
        U[k][k:] = A[k][k:]
        for i in range(k+1,n):
            L[i][k] = A[i][k]/U[k][k] 
            for j in range(k,n):
                A[i][j] -= L[i][k]*U[k][j]
    return L,U
n = int(input("Enter the size of matrix (n): "))
A = []
for i in range(n):
    row = []
    print(f"Enter elements of row {i+1}:")
    for j in range(n):
        val = float(input(f"Element[{i+1},{j+1}]: "))
        row.append(val)
    A.append(row)
L,U = LU(A)
print("matrix L",L)
print("matrix U",U)
print("matrix LU = A",np.dot(L,U))


# Newton forward and backward difference

x = [1, 3, 10, 11, 13]
y = [0.66, 2,6.67,7.33,8.66]


def forward_difference_table(x, y):
    n = len(x)
    diff_table = [y.copy()]
    for i in range(1, n):
        diff = []
        for j in range(n - i):
            value = diff_table[i - 1][j + 1] - diff_table[i - 1][j]
            diff.append(value)
        diff_table.append(diff)
    return diff_table

def backward_difference_table(x, y):
    n = len(x)
    diff_table = [y.copy()]
    for i in range(1, n):
        prev_row = diff_table[i - 1]
        diff = []
        for j in range(1, len(prev_row)):
            value = prev_row[j] - prev_row[j - 1]
            diff.append(value)
        diff_table.append(diff)
    return diff_table


def newton_forward(x, y, value):
    n = len(x)
    diff_table = forward_difference_table(x, y)
    h = x[1] - x[0]
    p = (value - x[0]) / h
    result = y[0]
    fact = 1
    p_prod = 1
    for i in range(1, n):
        p_prod *= (p - (i - 1))
        fact *= i
        result += (p_prod * diff_table[i][0]) / fact
    return result


def newton_backward(x, y, value):
    n = len(x)
    diff_table = backward_difference_table(x, y)
    h = x[1] - x[0]
    p = (value - x[-1]) / h
    result = y[-1]
    fact = 1
    p_prod = 1
    for i in range(1, n):
        p_prod *= (p + (i - 1))
        fact *= i
        result += (p_prod * diff_table[i][-1]) / fact  # <- fixed here
    return result


print("Forward:", newton_forward(x, y, 3.5))
print("Backward:", newton_backward(x, y, 4.5))
def divided_difference_table(x, y):
    n = len(x)
    table = [y.copy()]
    for i in range(1, n):
        prev_row = table[i-1]
        diff = [(prev_row[j+1] - prev_row[j]) / (x[j+i] - x[j]) for j in range(n-i)]
        table.append(diff)
    return table

def newton_divided_difference(x, y, value):
    table = divided_difference_table(x, y)
    n = len(x)
    result = table[0][0]
    for i in range(1, n):
        term = table[i][0]
        for j in range(i):
            term *= (value - x[j])
        result += term
    return result

print(newton_divided_difference(x,y,10.5))

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


import math as M
from scipy.integrate import dblquad  

def f(x, y):
    return x**2 + y**2

def simpson_integration_y(x, c, d, m):
    if m % 2 == 1:
        print("Number of subintervals (y) must be even.")
        return 
    
    hy = (d - c) / m
    y = c
    integral_y = f(x, y) + f(x, d)

    for j in range(1, m):
        y += hy
        if j % 2 == 0:
            integral_y += 2 * f(x, y)
        else:
            integral_y += 4 * f(x, y)

    integral_y *= hy / 3
    return integral_y
def simpson_double_integration(a, b, n, c, d, m):
    if n % 2 == 1:
        print("Number of subintervals (x) must be even.")
        return 
    
    hx = (b - a) / n
    x = a
    integral = simpson_integration_y(x, c, d, m) + simpson_integration_y(b, c, d, m)

    for i in range(1, n):
        x += hx
        if i % 2 == 0:
            integral += 2 * simpson_integration_y(x, c, d, m)
        else:
            integral += 4 * simpson_integration_y(x, c, d, m)

    integral *= hx / 3
    return integral

a = float(input("Enter lower limit for x: "))
b = float(input("Enter upper limit for x: "))
c = float(input("Enter lower limit for y: "))
d = float(input("Enter upper limit for y: "))
n = int(input("Enter number of intervals for x (even): "))
m = int(input("Enter number of intervals for y (even): "))


simpson_result = simpson_double_integration(a, b, n, c, d, m)
print("\nApproximate Double Integral (Simpson's Rule) =", simpson_result)

true_result, error = dblquad(lambda y, x: f(x, y), a, b, lambda x: c, lambda x: d)
print("Verification using SciPy dblquad =", true_result)
print("Error estimate =", error)


import numpy as np

def is_tridiagonal(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and matrix[i][j] != 0:
                return False
    return True

def gauss_elimination(a, b):
    n = len(b)
    for i in range(n):
        if a[i][i] == 0:
            for r in range(i + 1, n):
                if a[r][i] != 0:
                    for c in range(n):
                        a[i][c], a[r][c] = a[r][c], a[i][c]
                    b[i], b[r] = b[r], b[i]
                    break
            else:
                raise ValueError("Matrix is singular or nearly singular")

        pivot = a[i][i]

        # Normalize pivot row
        for k in range(i, n):
            a[i][k] = a[i][k] / pivot
        b[i] = b[i] / pivot

        # Eliminate below
        for j in range(i + 1, n):
            factor = a[j][i]
            for k in range(i, n):
                a[j][k] = a[j][k] - factor * a[i][k]
            b[j] = b[j] - factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] = x[i] - a[i][j] * x[j]
    return x


# Example matrix (you can change this)
# A = np.array([[2, -1, 0],
#               [-1, 2, -1],
#               [0, -1, 2]], dtype=float)   # ✅ This one IS tridiagonal

# b = np.array([1, 2, 3], dtype=float)
n = int(input("Enter the size of matrix (n): "))
A = []
for i in range(n):
    row = []
    print(f"Enter elements of row {i+1}:")
    for j in range(n):
        val = float(input(f"Element[{i+1},{j+1}]: "))
        row.append(val)
    A.append(row)

B = []
for i in range(n):
    val = float(input(f"Element for B[{i+1}] : "))
    B.append(val)

# Check first
if is_tridiagonal(A):
    sol = gauss_elimination(A.copy(), B.copy())
    print("Matrix is tridiagonal.") 
    for i in range(n):
        print("solution:", sol[i])
    # print("Solution:", sol)
else:
    print("Matrix is NOT tridiagonal. Cannot solve.")

import numpy as np
import matplotlib.pyplot as plt

# ----------------- TENNIS BALL (official values) -----------------
m   = 0.057                     # kg (official mass)
g   = 9.81
rho = 1.2
r   = 0.0335                     # radius = 3.35 cm
A   = np.pi * r**2               # exact area
Cd  = 0.5

y = 20.0                         # drop height
v = 0.0
t = 0.0
dt = 0.005                       # smaller step = perfect accuracy

time_list = [t]
height    = [y]
velocity  = [v]

while y > 0:
    # Save current state
    time_list.append(t)
    height.append(y)
    velocity.append(v)
    
    # FULL RK4 for BOTH position and velocity
    k1y = v
    k1v = -g - (0.5*rho*A*Cd*v*abs(v))/m

    k2y = v + 0.5*dt*k1v
    k2v = -g - (0.5*rho*A*Cd*k2y*abs(k2y))/m

    k3y = v + 0.5*dt*k2v
    k3v = -g - (0.5*rho*A*Cd*k3y*abs(k3y))/m

    k4y = v + dt*k3v
    k4v = -g - (0.5*rho*A*Cd*k4y*abs(k4y))/m

    v = v + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
    y = y + (dt/6)*(k1y + 2*k2y + 2*k3y + k4y)
    t = t + dt

    if y <= 0:
        height[-1] = 0
        break
v_terminal = np.sqrt(2*m*g / (rho*A*Cd))

print(f"Theoretical terminal velocity = {v_terminal:.1f} m/s")
print(f"Impact speed (simulation)      = {abs(v):.1f} m/s")
print(f"Time to ground                 = {t:.2f} s")
print(f"No-drag impact speed would be  = {np.sqrt(2*g*20):.1f} m/s")

# ----------------- PERFECT PLOT -----------------
plt.figure(figsize=(11,4.5))

plt.subplot(1,2,1)
plt.plot(time_list, velocity, 'orange', lw=3)
plt.axhline(-v_terminal, color='red', linestyle='--', linewidth=2, label=f'Terminal = {v_terminal:.1f} m/s')
plt.title("Tennis Ball Drop – Velocity", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("Velocity (m/s)")
plt.grid(alpha=0.3); plt.legend()

plt.subplot(1,2,2)
plt.plot(time_list, height, 'green', lw=3)
plt.title("Tennis Ball Drop – Height", fontsize=14)
plt.xlabel("Time (s)"); plt.ylabel("Height (m)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()