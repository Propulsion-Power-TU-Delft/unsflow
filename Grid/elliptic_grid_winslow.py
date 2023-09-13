import numpy as np
import matplotlib.pyplot as plt

# parameters
nx = 40
ny = 30
maxit = 2500
show = 1  # 1 for yes, 0 for no to display solution while solving
Ermax = 1e-5

# initializing the borders
f = lambda x: x ** 2
c1 = np.vstack((-np.ones(ny), np.linspace(0, 4, ny)))  # left
c2 = np.vstack((np.linspace(-1, 0, nx), np.zeros(nx)))  # bottom
c3 = np.vstack((np.linspace(0, 2, ny), f(np.linspace(0, 2, ny))))  # right
c4 = np.vstack((np.linspace(-1, 2, nx), 4 * np.ones(nx)))  # top

# plot the border
plt.figure()
plt.plot(c1[0, :], c1[1, :], label='c1')
plt.plot(c2[0, :], c2[1, :], label='c2')
plt.plot(c3[0, :], c3[1, :], label='c3')
plt.plot(c4[0, :], c4[1, :], label='c4')
plt.legend()

alpha = np.zeros((nx, ny))
beta = np.zeros((nx, ny))
gamma = np.zeros((nx, ny))

# initialize the X grid
X = np.zeros((nx, ny))
X[0, :] = c1[0, :]  # left
X[-1, :] = c3[0, :]  # right
X[:, -1] = c4[0, :]  # top
X[:, 0] = c2[0, :]  # bottom
xi = np.linspace(0, 1, nx)
dxi = xi[1] - xi[0]

# initialize the Y grid
Y = np.zeros((nx, ny))
Y[0, :] = c1[1, :]  # left
Y[-1, :] = c3[1, :]  # right
Y[:, -1] = c4[1, :]  # top
Y[:, 0] = c2[1, :]  # bottom
eta = np.linspace(0, 1, ny)
deta = eta[1] - eta[0]

newX = X.copy()
newY = Y.copy()

Er1 = np.zeros(maxit)
Er2 = np.zeros(maxit)

g11 = np.zeros((nx, ny))
g12 = np.zeros((nx, ny))
g22 = np.zeros((nx, ny))

a = np.zeros((nx, ny))
b = np.zeros((nx, ny))
c = np.zeros((nx, ny))
d = np.zeros((nx, ny))
e = np.zeros((nx, ny))

# internal slices of the matrices
i = slice(1, nx - 1)
ip = slice(2, nx)
im = slice(0, nx - 2)
j = slice(1, ny - 1)
jp = slice(2, ny)
jm = slice(0, ny - 2)

g11[i, j] = ((X[ip, j] - X[im, j]) / 2 / dxi) ** 2 + ((Y[ip, j] - Y[im, j]) / 2 / dxi) ** 2

g22[i, j] = ((X[i, jp] - X[i, jm]) / 2 / deta) ** 2 + ((Y[i, jp] - Y[i, jm]) / 2 / deta) ** 2

g12[i, j] = ((X[ip, j] - X[im, j]) / 2 / dxi) * ((X[i, jp] - X[i, jm]) / 2 / deta) + \
            ((Y[ip, j] - Y[im, j]) / 2 / dxi) * ((Y[i, jp] - Y[i, jm]) / 2 / deta)

a[i, j] = g22[i, j] / dxi ** 2

b[i, j] = 2 * g22[i, j] / dxi ** 2 + 2 * g11[i, j] / deta ** 2

c[i, j] = a[i, j]

d[i, j] = g11[i, j] / deta ** 2 * (X[i, jp] + X[i, jm]) - 2 * g12[i, j] * (
            X[ip, jp] + X[im, jm] - X[im, jp] - X[ip, jm]) / 4 / dxi / deta

e[i, j] = g11[i, j] / deta ** 2 * (Y[i, jp] + Y[i, jm]) - 2 * g12[i, j] * (
            Y[ip, jp] + Y[im, jm] - Y[im, jp] - Y[ip, jm]) / 4 / dxi / deta

# A = np.zeros((nx, ny))
# for ii in range(1, nx-1):
#     for jj in range(1, ny-1):
#         A[i, j]






print('check')
