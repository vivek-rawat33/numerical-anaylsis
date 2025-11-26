import math as M
def simpson_integration(f, a, b, n):
    if n % 2 == 1:
       print("Number of subintervals must be even.")
       return 

    h = (b - a) / n
    x = a
    integral = f(x) + f(b)

    for i in range(1, n):
        x += h
        if i % 2 == 0:
            integral += 2 * f(x)
        else:
            integral += 4 * f(x)

    integral *= h / 3
    return integral
def example_function(x):  
    return M.log(M.sin(x))
a =float(input("enter the lower limit :"))
b =float(input("enter the upper limit :"))
n = int(input("enter the number of iterations :"))
result = simpson_integration(example_function, a, b, n)
print("The integral is:", result)