import numpy as np
from scipy.sparse.linalg import cg


def conjugate_grad(A, b, x_init):
    r = b - np.dot(A, x_init)
    p = r
    r_norm = np.dot(r, r)
    x = x_init
    for i in range(5000):
        Ap = np.dot(A, p)
        alpha = r_norm / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_new_norm = np.dot(r, r)
        beta = r_new_norm / r_norm
        r_norm = r_new_norm
        if r_new_norm < 1e-5:
            return x, i + 1
        p = r + beta * p
    return x, i + 1


n = 1000

A = np.array([[1, 2, 4, 8], [1, 3, 9, 27], [1, 4, 16, 64], [1, 5, 25, 125]], dtype='float')
b = np.array([1, -1, 1, -1], dtype='float')
x_init = np.zeros(len(b))
x, iter = conjugate_grad(np.dot(A.T, A), np.dot(A.T, b), x_init)
print('x:', x)
print('iter:', iter)
print('computed b:', np.dot(A, x))
print('real b:', b)

x_init = np.ones(len(b))
x, iter = conjugate_grad(np.dot(A.T, A), np.dot(A.T, b), x_init)
print('x:', x)
print('iter:', iter)
print('computed b:', np.dot(A, x))
print('real b:', b)

# x, iter = cg(np.dot(A.T,A),np.dot(A.T,b), x_init, maxiter=5000)
# print('x:', x)
# print('iter:', iter)
# print('computed b:', np.dot(A, x))
# print('real b:', b)
