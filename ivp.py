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
