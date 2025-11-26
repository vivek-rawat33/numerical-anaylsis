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
