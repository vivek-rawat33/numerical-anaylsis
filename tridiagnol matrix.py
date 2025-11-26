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
