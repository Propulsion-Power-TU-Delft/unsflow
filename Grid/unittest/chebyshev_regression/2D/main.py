import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander2d
from numpy.linalg import lstsq
from Grid.src.functions import create_folder

X_MIN = -0.1
X_MAX = 1
Y_MIN = 1
Y_MAX = 1.5
N_SAMPLES = 20
SIGMA_ERROR = 1e-3
DEGREE = 10
TEST_FUNCTIONS = np.linspace(1, 5, 5)


def func(x, y):
    if TEST_FUNCTION == 0:
        return x + y
    elif TEST_FUNCTION == 1:
        return x ** 2 - y ** 2
    elif TEST_FUNCTION == 2:
        return np.sin(3 * x) - np.cos(2 * y)
    elif TEST_FUNCTION == 3:
        return np.sin(4 * x * y)
    elif TEST_FUNCTION == 4:
        return np.sin(4 * x * y) - x ** 2 / y ** 3
    elif TEST_FUNCTION == 5:
        return x ** 2 - np.sin(3 * x * y) / y ** 3


def func_primex(x, y):
    if TEST_FUNCTION == 0:
        return np.zeros_like(x) + 1
    elif TEST_FUNCTION == 1:
        return 2 * x
    elif TEST_FUNCTION == 2:
        return 3 * np.cos(3 * x)
    elif TEST_FUNCTION == 3:
        return np.cos(4 * x * y) * 4 * y
    elif TEST_FUNCTION == 4:
        return np.cos(4 * x * y) * 4 * y - 2 * x / y ** 3
    elif TEST_FUNCTION == 5:
        return 2 * x - (np.cos(3 * x * y) * 3) / y ** 2


def func_primey(x, y):
    if TEST_FUNCTION == 0:
        return np.zeros_like(x) + 1
    elif TEST_FUNCTION == 1:
        return - 2 * y
    elif TEST_FUNCTION == 2:
        return 2 * np.sin(2 * y)
    elif TEST_FUNCTION == 3:
        return np.cos(4 * x * y) * 4 * x
    elif TEST_FUNCTION == 4:
        return np.cos(4 * x * y) * 4 * x + 3 * x ** 2 * y ** (-4)
    elif TEST_FUNCTION == 5:
        return 3 * (np.sin(3 * x * y) - x * y * np.cos(3 * x * y))


for TEST_FUNCTION in TEST_FUNCTIONS:
    folder = 'test_function_%02i' % (TEST_FUNCTION)
    create_folder(folder)

    # Generate sample data
    np.random.seed(0)
    x_sample = np.linspace(X_MIN, X_MAX, N_SAMPLES)
    y_sample = np.linspace(Y_MIN, Y_MAX, N_SAMPLES)
    Y_sample, X_sample = np.meshgrid(y_sample, x_sample)
    Z_sample = func(X_sample, Y_sample)

    # Perform Chebyshev regression
    A = chebvander2d(X_sample.flatten(), Y_sample.flatten(), [DEGREE, DEGREE])
    coeffs, _, _, _ = lstsq(A, Z_sample.flatten(), rcond=None)

    # Generate points to plot the regression surface
    A = chebvander2d(X_sample.flatten(), Y_sample.flatten(), [DEGREE, DEGREE])
    Z_regr = np.dot(A, coeffs).reshape(X_sample.shape)

    plt.figure()
    plt.contourf(X_sample, Y_sample, Z_sample, levels=25)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$f$')
    plt.colorbar()
    plt.savefig(folder + '/z.pdf', bbox_inches='tight')

    plt.figure()
    plt.contourf(X_sample, Y_sample, (Z_regr - Z_sample) / np.linalg.norm(Z_sample, ord=2), levels=25)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$\hat{\varepsilon}$')
    plt.colorbar()
    plt.savefig(folder + '/z_err.pdf', bbox_inches='tight')


    def chebyshev_polynomial(k, x):
        """
        Definition by recursion.
        :param k: order of the polynomial
        :param x: evaluation points
        """
        if k == 0:
            return np.ones_like(x)
        elif k == 1:
            return x
        else:
            return 2 * x * chebyshev_polynomial(k - 1, x) - chebyshev_polynomial(k - 2, x)


    def chebyshev_derivative_recursive(k, x):
        """
        Definition by recursion.
        :param k: order of the polynomial
        :param x: evaluation points
        """
        if k == 0:
            return np.zeros_like(x)
        elif k == 1:
            return np.ones_like(x)
        else:
            return 2 * chebyshev_polynomial(k - 1, x) + 2 * x * chebyshev_derivative_recursive(k - 1,
                                                                                               x) - chebyshev_derivative_recursive(
                k - 2, x)


    def compute_derivative_matrices_recursive(degrees, x, y):
        """
        Build the Van Der Monde matrices of the x and y derivatives.
        :param degrees: (x,y) degrees
        :param x: evaluation points
        :param y: evaluation points
        """
        n_samples = len(x)
        DX = np.zeros((n_samples, (degrees[0] + 1) * (degrees[1] + 1)))
        DY = np.zeros((n_samples, (degrees[0] + 1) * (degrees[1] + 1)))
        for i in range(degrees[0] + 1):
            for j in range(degrees[1] + 1):
                k = j + i * (degrees[0] + 1)
                DX[:, k] = chebyshev_derivative_recursive(i, x) * chebyshev_polynomial(j, y)
                DY[:, k] = chebyshev_polynomial(i, x) * chebyshev_derivative_recursive(j, y)
        return DX, DY


    B, C = compute_derivative_matrices_recursive([DEGREE, DEGREE], X_sample.flatten(), Y_sample.flatten())

    ZX_regr = np.dot(B, coeffs).reshape(X_sample.shape)
    ZY_regr = np.dot(C, coeffs).reshape(X_sample.shape)

    plt.figure()
    plt.contourf(X_sample, Y_sample, func_primex(X_sample, Y_sample), levels=25)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$f_x$')
    plt.colorbar()
    plt.savefig(folder + '/dzdx.pdf', bbox_inches='tight')

    plt.figure()
    plt.contourf(X_sample, Y_sample, func_primey(X_sample, Y_sample), levels=25)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$f_y$')
    plt.colorbar()
    plt.savefig(folder + '/dzdy.pdf', bbox_inches='tight')

    fx = func_primex(X_sample, Y_sample)
    fy = func_primey(X_sample, Y_sample)

    plt.figure()
    plt.contourf(X_sample, Y_sample, (ZX_regr - fx) / np.linalg.norm(fx, ord=2), levels=25)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$\hat{\varepsilon_x}$')
    plt.colorbar()
    plt.savefig(folder + '/dzdx_err.pdf', bbox_inches='tight')

    plt.figure()
    plt.contourf(X_sample, Y_sample, (ZY_regr - fy) / np.linalg.norm(fy, ord=2), levels=25)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$\hat{\varepsilon_y}$')
    plt.colorbar()
    plt.savefig(folder + '/dzdy_err.pdf', bbox_inches='tight')  # plt.show()
