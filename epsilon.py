import numpy as np


def epsilon(A, b, ep, x_init, max_iter=100000000):
    x = x_init
    for i in range(0, max_iter):
        x_new = x + ep * (np.dot(np.dot(A.T, A), x) - np.dot(A.T, b))
        if np.linalg.norm(x_new - x) < 1e-8:
            return x_new, i + 1
        x = x_new
    return x, i + 1


A = np.array([[1, 2, 4, 8], [1, 3, 9, 27], [1, 4, 16, 64], [1, 5, 25, 125]], dtype='float')
b = np.array([1, -1, 1, -1], dtype='float')

x_init = np.zeros(len(b))
x, iter = epsilon(A, b, -8e-5, x_init)

print('x:', x)
print('iter:', iter)
print('computed b:', np.dot(A, x))
print('real b:', b)

x_init = np.ones(len(b))
x, iter = epsilon(A, b, -8e-5, x_init)

print('x:', x)
print('iter:', iter)
print('computed b:', np.dot(A, x))
print('real b:', b)
