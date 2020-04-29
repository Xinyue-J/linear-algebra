import numpy as np
from scipy.sparse.linalg import expm
from sympy import *

A = np.array([[1, 2, 4, 8], [1, 3, 9, 27], [1, 4, 16, 64], [1, 5, 25, 125]], dtype='float')
d = np.array([[4, 0, 0, 0], [0, 14, 0, 0], [0, 0, 54, 0], [0, 0, 0, 224]], dtype='float')

B = np.dot(A, np.linalg.inv(d))
print(B)
value, vector = np.linalg.eig(B)
print('eigenvalue of B:', value, 'eigenvector of B:', vector)
value2, vector2 = np.linalg.eig(B.T)
print('eigenvalue of B^T:', value2, 'eigenvector of B^T:', vector2)
det = np.linalg.det(B)
print('determinant:', det)

t = symbols('t')
dia=np.diag(value)
exp=np.diag(np.exp(value)**t)
final=np.dot(np.dot(vector,exp),np.linalg.inv(vector))
print('exp（At):',final)

q, r = np.linalg.qr(B)
print('Q:', q, )
print('R:', r)
u, sigma, vT = np.linalg.svd(B)
print('U:', u)
print('sigma:', sigma)
print('V:', vT.T)


l1 = np.linalg.cholesky(np.dot(B.T, B))
l2 = np.linalg.cholesky(np.dot(B, B.T))
print('Cholesky factorization of BTB:', l1)
print('Cholesky factorization of BBT:', l2)
