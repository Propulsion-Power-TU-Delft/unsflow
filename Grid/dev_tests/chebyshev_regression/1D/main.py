import numpy as np
import matplotlib.pyplot as plt

X_MIN = 0.1
X_MAX = 2
N_SAMPLES = 10
SIGMA_ERROR = 1e-2
DEGREE = 7



# Define the function to regress
def func(x):
    return x**2-np.sin(6*x)/x

def func_prime(x):
    return 2*x-(6*x*np.cos(6*x)-np.sin(6*x))/x**2

# Generate sample data
np.random.seed(0)
x_samples = np.linspace(X_MIN, X_MAX, N_SAMPLES)
y_samples = func(x_samples) + np.random.normal(0, SIGMA_ERROR, N_SAMPLES)  # Adding some noise

# Perform Chebyshev regression
A = np.polynomial.chebyshev.chebvander(x_samples, DEGREE)
coeffs = np.linalg.lstsq(A, y_samples, rcond=None)[0]

# Generate points to plot the regression curve
x_plot = np.linspace(X_MIN, X_MAX, 100)
A_plot = np.polynomial.chebyshev.chebvander(x_plot, DEGREE)
y_plot = np.dot(A_plot, coeffs)


# Plot the original function and the regression curve
plt.figure()
plt.plot(x_plot, func(x_plot), label='Analytical')
plt.plot(x_plot, y_plot, label='Chebyshev Regression %i' %(DEGREE), linestyle='--')
plt.scatter(x_samples, y_samples, color='red', label='Sample Points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Chebyshev Regression')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig('y.pdf', bbox_inches='tight')






def chebyshev_polynomial(k, x):
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return x
    else:
        return 2 * x * chebyshev_polynomial(k-1, x) - chebyshev_polynomial(k-2, x)
# Function to compute the derivative of the Chebyshev polynomial of degree k
def chebyshev_derivative_recursive(k, x):
    if k == 0:
        return np.zeros_like(x)
    elif k == 1:
        return np.ones_like(x)
    else:
        return 2 * chebyshev_polynomial(k-1, x) + 2 * x * chebyshev_derivative_recursive(k-1, x) - chebyshev_derivative_recursive(k-2, x)

# Compute the derivative matrix B
def compute_derivative_matrix_recursive(degree, x_samples):
    n_samples = len(x_samples)
    B = np.zeros((n_samples, degree + 1))
    for k in range(degree + 1):
        B[:, k] = chebyshev_derivative_recursive(k, x_samples)
    return B

B_plot = compute_derivative_matrix_recursive(DEGREE, x_plot)
y_prime_plot = np.dot(B_plot, coeffs)

# Plot the original function and the regression curve
plt.figure()
plt.plot(x_plot, func_prime(x_plot), label='Analytical')
plt.plot(x_plot, y_prime_plot, label='Chebyshev Regression %i' %(DEGREE), linestyle='--')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.title('Regression based derivative')
plt.grid(alpha=0.2)
plt.savefig('dydx.pdf', bbox_inches='tight')
plt.show()