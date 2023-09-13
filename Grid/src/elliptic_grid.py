import numpy as np
import matplotlib.pyplot as plt

# parameters
nx = 40
ny = 30
maxit = 2500
show = 0  # 1 for yes, 0 for no to display solution while solving
Ermax = 1e-5

# initializing the borders
f = lambda x: x**2
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
X[0, :] = c3[0, :]  # right
X[-1, :] = c1[0, :]  # left
X[:, -1] = np.flip(c4[0, :])  # top
X[:, 0] = np.flip(c2[0, :])  # bottom

# initialize the Y grid
Y = np.zeros((nx, ny))
Y[0, :] = c3[1, :]  # right
Y[-1, :] = c1[1, :]  # left
Y[:, -1] = np.flip(c4[1, :])  # top
Y[:, 0] = np.flip(c2[1, :])  # bottom

newX = X.copy()
newY = Y.copy()

Er1 = np.zeros(maxit)
Er2 = np.zeros(maxit)

# calculating by iterations
for t in range(maxit):

    # prepare the slices of the 2D array, to update internal points
    i = slice(1, nx-1)
    i_plus = slice(2, nx)
    i_minus = slice(0, nx - 2)
    j = slice(1, ny-1)
    j_plus = slice(2, ny)
    j_minus = slice(0, ny-2)

    alpha[i, j] = (1 / 4) * ((X[i, j_plus] - X[i, j_minus]) ** 2 + (Y[i, j_plus] - Y[i, j_minus]) ** 2)

    beta[i, j] = (1 / 16) * ((X[i_plus, j] - X[i_minus, j]) * (X[i, j_plus] - X[i, j_minus]) +
                             (Y[i_plus, j] - Y[i_minus, j]) * (Y[i, j_plus] - Y[i, j_minus]))

    gamma[i, j] = (1 / 4) * ((X[i_plus, j] - X[i_minus, j]) ** 2 + (Y[i_plus, j] - Y[i_minus, j]) ** 2)

    newX[i, j] = ((-0.5) / (alpha[i, j] + gamma[i, j] + 1e-9)) * (
            2 * beta[i, j] * (X[i_plus, j_plus] - X[i_minus, j_plus] - X[i_plus, j_minus] + X[i_minus, j_minus]) -
            alpha[i, j] * (X[i_plus, j] + X[i_minus, j]) - gamma[i, j] * (X[i, j_plus] + X[i, j_minus]))

    newY[i, j] = ((-0.5) / (alpha[i, j] + gamma[i, j] + 1e-9)) * (
            2 * beta[i, j] * (Y[i_plus, j_plus] - Y[i_minus, j_plus] - Y[i_plus, j_minus] + Y[i_minus, j_minus]) -
            alpha[i, j] * (Y[i_plus, j] + Y[i_minus, j]) - gamma[i, j] * (Y[i, j_plus] + Y[i, j_minus]))

    Er1[t] = np.max(np.abs(newX - X))
    Er2[t] = np.max(np.abs(newY - Y))

    # Neuman BC
    newY[-1, :] = newY[-2, :]  # left

    # don't use the following line, just an example to show that b.c are tricky
    # newY[0, :] = newY[1, :]  # right

    X = newX.copy()
    Y = newY.copy()

    if Er1[t] < Ermax and Er2[t] < Ermax:
        break

    if show == 1:
        if t % 10 == 0:
            plt.clf()
            plt.axis('equal')
            for m in range(nx):
                plt.plot(X[m, :], Y[m, :], 'b', lw = 0.1)
            for m in range(ny):
                plt.plot(X[:, m], Y[:, m], 'b', lw= 0.1)
            plt.pause(0.001)

if t == maxit:
    print('Convergence not reached')

plt.clf()
plt.axis('equal')
for m in range(nx):
    plt.plot(X[m, :], Y[m, :], 'b')
for m in range(ny):
    plt.plot(X[:, m], Y[:, m], color=[0, 0, 0])
plt.show()
