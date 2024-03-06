import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, tan, sqrt, meshgrid
import Sun
from Utils.styles import *


def function2D(X, Y, func):
    if func == 0:
        f = (X ** 2 + 2 * Y - 3 * np.sin(5 * X * Y) + Y ** 3)
        dfdx = 2 * X - 3 * np.cos(5 * X * Y) * 5 * Y
        dfdy = 2 - 3 * np.cos(5 * X * Y) * 5 * X + 3 * Y ** 2
        df2dx2 = np.ones_like(X) * 2 + 3 * np.sin(5 * X * Y) * 25 * Y ** 2
        df2dy2 = 3 * sin(5 * X * Y) * 25 * X ** 2 + 6 * Y
        df2dxdy = 75 * X * Y * sin(5 * X * Y) - 15 * cos(5 * X * Y)
        name_str = r'$x^2+2y-3 \sin(5xy) + y^3$'
    elif func == 1:
        f = X ** 2 - Y ** 2
        dfdx = 2 * X
        dfdy = -2 * Y
        df2dx2 = np.ones_like(X) * 2
        df2dy2 = -np.ones_like(X) * 2
        df2dxdy = np.zeros_like(X)
        name_str = r'$x^2-y^2$'
    elif func == 2:
        f = sin(X) + cos(Y)
        dfdx = cos(X)
        dfdy = -sin(Y)
        df2dx2 = -sin(X)
        df2dy2 = -cos(Y)
        df2dxdy = np.zeros_like(X)
        name_str = r'$\sin{x} + \cos{y}$'
    return f, dfdx, dfdy, df2dx2, df2dy2, df2dxdy, name_str


plots = False
NODES = np.linspace(10, 50, 5, dtype=int)
FUNC_NUM = 0
eps_x = []
eps_y = []
eps_xx = []
eps_yy = []
eps_xy = []
for i in range(len(NODES)):
    print('Nodes: %i' % NODES[i])
    NX = NODES[i]
    NY = NODES[i]

    x_gl = Sun.src.general_functions.GaussLobattoPoints(NX)
    y_gl = Sun.src.general_functions.GaussLobattoPoints(NY)

    X, Y = meshgrid(x_gl, y_gl, indexing='ij')
    Z, dZdx, dZdy, dZ2dX2, dZ2dy2, dZ2dxdy, name_func = function2D(X, Y, func=FUNC_NUM)

    if plots:
        plt.figure()
        plt.contourf(X, Y, Z)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        plt.title(name_func)
        plt.colorbar()
        plt.savefig('pictures/func_%i_nodes_%i_%i.pdf' % (FUNC_NUM, NX, NY), bbox_inches='tight')


    Dx = Sun.src.general_functions.ChebyshevDerivativeMatrixBayliss(
        x_gl)  # derivative operator in xi, Bayliss formulation
    Dy = Sun.src.general_functions.ChebyshevDerivativeMatrixBayliss(y_gl)

    # First (serial implementation of the spectral differentiation matrix in 2D)
    # dZdX_gl = np.zeros_like(Z)
    # dZdY_gl = np.zeros_like(Z)
    # for i in range(0, NX):
    #     for j in range(0, NY):
    #         tmpx = 0
    #         tmpy = 0
    #         for nx in range(0, NX):
    #             tmpx += Dx[i, nx] * Z[nx, j]
    #         for ny in range(0, NY):
    #             tmpy += Dy[j, ny] * Z[i, ny]
    #         dZdX_gl[i, j] = tmpx
    #         dZdY_gl[i, j] = tmpy

    # plt.figure()
    # plt.contourf(X, Y, dZdX_gl)
    # plt.xlabel(r'$\xi$')
    # plt.ylabel(r'$\eta$')
    # plt.title(r'$f_x(x,y)$ GL SER')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.contourf(X, Y, dZdY_gl)
    # plt.title(r'$f_y(x,y)$ GL SER')
    # plt.xlabel(r'$\xi$')
    # plt.ylabel(r'$\eta$')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.contourf(X, Y, dZdX_gl-dZdx)
    # plt.xlabel(r'$\xi$')
    # plt.ylabel(r'$\eta$')
    # plt.title(r'$\varepsilon_x (x,y)$ GL SER')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.contourf(X, Y, dZdY_gl-dZdy)
    # plt.title(r'$\varepsilon_y (x,y)$ GL SER')
    # plt.xlabel(r'$\xi$')
    # plt.ylabel(r'$\eta$')
    # plt.colorbar()

    # Second (parallel) implementation of the spectral differentiation method (Trefethen book)
    Ix = np.eye(Z.shape[0])
    Iy = np.eye(Z.shape[1])
    Dx_mat = np.kron(Dx, Iy)
    Dy_mat = np.kron(Ix, Dy)
    Dx_mat2 = np.kron(Dx @ Dx, Iy)
    Dy_mat2 = np.kron(Ix, Dy @ Dy)
    DxDy_mat = Dx_mat @ Dy_mat

    dZdX_gl_mat = np.reshape(Dx_mat @ Z.flatten(), (NX, NY))
    dZdY_gl_mat = np.reshape(Dy_mat @ Z.flatten(), (NX, NY))

    dZ2dX2_gl_mat = np.reshape(Dx_mat2 @ Z.flatten(), (NX, NY))
    dZ2dY2_gl_mat = np.reshape(Dy_mat2 @ Z.flatten(), (NX, NY))
    dZ2dxdy_gl_mat = np.reshape(DxDy_mat @ Z.flatten(), (NX, NY))

    if plots:
        plt.figure()
        plt.contourf(X, Y, dZdX_gl_mat - dZdx)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        plt.title(r'$\varepsilon_x (x,y)$')
        plt.colorbar()
        plt.savefig('pictures/func_%i_nodes_%i_%i_epsX.pdf' %(FUNC_NUM, NX, NY), bbox_inches='tight')

        plt.figure()
        plt.contourf(X, Y, dZdY_gl_mat - dZdy)
        plt.title(r'$\varepsilon_y (x,y)$')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        plt.colorbar()
        plt.savefig('pictures/func_%i_nodes_%i_%i_epsY.pdf' % (FUNC_NUM, NX, NY), bbox_inches='tight')

        plt.figure()
        plt.contourf(X, Y, dZ2dX2_gl_mat - dZ2dX2)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        plt.title(r'$\varepsilon_{xx} (x,y)$')
        plt.colorbar()
        plt.savefig('pictures/func_%i_nodes_%i_%i_epsXX.pdf' % (FUNC_NUM, NX, NY), bbox_inches='tight')

        plt.figure()
        plt.contourf(X, Y, dZ2dY2_gl_mat - dZ2dy2)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        plt.title(r'$\varepsilon_{yy} (x,y)$')
        plt.colorbar()
        plt.savefig('pictures/func_%i_nodes_%i_%i_epsYY.pdf' % (FUNC_NUM, NX, NY), bbox_inches='tight')

        plt.figure()
        plt.contourf(X, Y, dZ2dxdy_gl_mat - dZ2dxdy)
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\eta$')
        plt.title(r'$\varepsilon_{xy} (x,y)$')
        plt.colorbar()
        plt.savefig('pictures/func_%i_nodes_%i_%i_epsXY.pdf' % (FUNC_NUM, NX, NY), bbox_inches='tight')

    eps_x.append(np.max(np.abs(dZdX_gl_mat - dZdx)))
    eps_y.append(np.max(np.abs(dZdY_gl_mat - dZdy)))
    eps_xx.append(np.max(np.abs(dZ2dX2_gl_mat - dZ2dX2)))
    eps_yy.append(np.max(np.abs(dZ2dY2_gl_mat - dZ2dy2)))
    eps_xy.append(np.max(np.abs(dZ2dxdy_gl_mat - dZ2dxdy)))

eps_x = np.array(eps_x)
eps_y = np.array(eps_y)
eps_xx = np.array(eps_xx)
eps_yy = np.array(eps_yy)
eps_xy = np.array(eps_xy)

plt.figure()
plt.plot(NODES, eps_x / eps_x[0], '-o', label=r'$\varepsilon_x$')
plt.plot(NODES, eps_y / eps_y[0], '-o', label=r'$\varepsilon_y$')
plt.plot(NODES, eps_xx / eps_xx[0], '-o', label=r'$\varepsilon_{yy}$')
plt.plot(NODES, eps_yy / eps_yy[0], '-o', label=r'$\varepsilon_{yy}$')
plt.plot(NODES, eps_xy / eps_xy[0], '-o', label=r'$\varepsilon_{xy}$')
plt.yscale('log')
plt.title('Scaled Errors')
plt.grid(alpha=grid_opacity)
plt.xlabel('nodes')
plt.ylabel(r'max $|\varepsilon|/\varepsilon_{0}$')
plt.savefig('pictures/func_%i_scaled_max_errors.pdf' % (FUNC_NUM), bbox_inches='tight')
plt.legend()

plt.figure()
plt.plot(NODES, eps_x, '-o', label=r'$\varepsilon_x$')
plt.plot(NODES, eps_y, '-o', label=r'$\varepsilon_y$')
plt.plot(NODES, eps_xx, '-o', label=r'$\varepsilon_{yy}$')
plt.plot(NODES, eps_yy, '-o', label=r'$\varepsilon_{yy}$')
plt.plot(NODES, eps_xy, '-o', label=r'$\varepsilon_{xy}$')
plt.yscale('log')
plt.legend()
plt.ylabel(r'max $|\varepsilon|$')
plt.xlabel('nodes')
plt.title('Absolute Errors')
plt.grid(alpha=grid_opacity)
plt.savefig('pictures/func_%i_absolute_max_errors.pdf' % (FUNC_NUM), bbox_inches='tight')


plt.show()
