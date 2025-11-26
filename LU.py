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