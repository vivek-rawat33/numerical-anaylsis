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

