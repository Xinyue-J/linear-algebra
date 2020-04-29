import numpy as np

A = np.array([[1, 2, 4, 8], [1, 3, 9, 27], [1, 4, 16, 64], [1, 5, 25, 125]], dtype='float')
b = np.array([1, -1, 1, -1], dtype='float')

Ab = np.hstack([A, b.reshape(-1, 1)])
n = len(b)

for i in range(n):
    a = Ab[i]
    for j in range(i + 1, n):
        b = Ab[j]
        m = a[i] / b[i]
        Ab[j] = a - m * b

for i in range(n - 1, -1, -1):
    Ab[i] = Ab[i] / Ab[i, i]
    a = Ab[i]
    for j in range(i - 1, -1, -1):
        b = Ab[j]
        m = b[i] / a[i]
        Ab[j] = b - m * a  # a - m * b

x = Ab[:, n]
print(x)
