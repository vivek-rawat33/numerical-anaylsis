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