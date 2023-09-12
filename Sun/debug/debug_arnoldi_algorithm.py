# test and debug the arnoldi algorithm to find the eigenvalues of a generalized EVP. Based on pag. 361 Strang Algebra book

import numpy as np
from scipy.sparse.linalg import eigs
import numpy as np
import matplotlib.pyplot as plt

# matrices of the generalized EVP A*x = lambda*M*x
dim = 300  # dimension of the matrix

real_part = np.random.uniform(-1, 1, dim)
imaginary_part = np.random.uniform(-1, 1, dim)
random_complex_array = real_part + 1j * imaginary_part

eigs_analytical = random_complex_array/2
A = np.diag(random_complex_array)

M = np.eye(dim)*2

# Compute eigenvalues and eigenvectors using ARPACK (Arnoldi algorithm). M is the RHS matrix, k is the numer of eigs to retrieve
# with corresponding eigenvalues
sigma = 0
k = 200
eig_val, eig_vec = eigs(A, M=M, k=k, sigma=sigma)

print("Eigenvalues:")
print(eig_val)
print("Eigenvectors:")
print(eig_vec)

# plot the determinant value
marker_size = 100
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(eigs_analytical.real, eigs_analytical.imag, marker='o', facecolors='none', edgecolors='black',
           s=marker_size, label=r'analytical')
ax.scatter(eig_val.real, eig_val.imag, marker='x', facecolors='black', label=r'numerical', s=marker_size)
ax.scatter(sigma.real, sigma.imag, marker='s', facecolors='red', edgecolors='red', label=r'$\sigma$ initial guess', s=marker_size)
ax.set_xlabel(r'$\lambda_{R}$')
ax.set_ylabel(r'$\lambda_{I}$')
ax.set_title(r'Arnoldi Algorithm Eigenvalues: $(N=%d, \ k=%d)$' %(dim, k))
ax.legend()
# fig.savefig('pictures/lambda_roots_m%d.pdf' %(m), bbox_inches='tight')
plt.show()