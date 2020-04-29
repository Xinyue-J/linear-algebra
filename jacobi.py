import numpy as np


def jacobi(A, b, x_init, epsilon=1e-5, max_iter=1000):
    D = np.diag(np.diag(A))
    LU = A - D
    x = x_init
    for i in range(0, max_iter):
        x_new = np.dot(np.linalg.inv(D), b - np.dot(LU, x))
        if np.linalg.norm(x_new - x) < epsilon:
            return x_new, i+1
        x = x_new
    return x, i+1


A = np.array([[1, 2, 4, 8], [1, 3, 9, 27], [1, 4, 16, 64], [1, 5, 25, 125]], dtype='float')
b = np.array([1, -1, 1, -1], dtype='float')

x_init = np.zeros(len(b))
x, iter = jacobi(A, b, x_init)

print('x:', x)
print('iter:', iter)
print('computed b:', np.dot(A, x))
print('real b:', b)

x_init = np.ones(len(b))
x, iter = jacobi(A, b, x_init)

print('x:', x)
print('iter:', iter)
print('computed b:', np.dot(A, x))
print('real b:', b)

