# Gauss-Seidel Method until steady (3 decimal places)
def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off_diag = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off_diag:
            return False
    return True

def gauss_seidel(a, b):
    n = len(a)
    x = [0.0] * n   # initial guess inside function
    iteration = 0
    if is_diagonally_dominant(a):
        while True:
            iteration += 1
            x_old = x.copy()
            for i in range(n):
                s = sum(a[i][j] * x[j] for j in range(n) if j != i)
                x[i] = (b[i] - s) / a[i][i]

        # stop if steady (difference < 0.001 → 3 decimals)
            if all(abs(x[i] - x_old[i]) < 0.001 for i in range(n)):
                break
        return [float(f"{val:.4f}") for val in x], iteration
    else:
        return "matrix is not diagonal dominant",0
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
