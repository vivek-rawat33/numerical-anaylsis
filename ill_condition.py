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
