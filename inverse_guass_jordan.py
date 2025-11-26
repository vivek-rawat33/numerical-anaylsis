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
