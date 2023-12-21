import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg import splu
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
from scipy.linalg import pinv
from numpy.polynomial import Polynomial

from scipy.optimize import curve_fit


def relative_residual(A, x, f):
    residual_norm = np.linalg.norm(A.dot(x) - f)
    f_norm = np.linalg.norm(f)
    return residual_norm / f_norm
def block_kaczmarz(A, f, K, max_iter=1000):
    x = np.zeros(A.shape[1])
    A_dense = A.toarray()  # Convert to dense array
    for _ in range(max_iter):
        for i in range(0, A.shape[0], K):
            A_block = A_dense[i:i+K, :]
            f_block = f[i:i+K]
            x += np.dot(A_block.T, np.linalg.solve(A_block.dot(A_block.T), f_block - A_block.dot(x)))
    return x

def block_cimmino(A, f, K, max_iter=1000):
    x = np.zeros(A.shape[1])
    A_dense = A.toarray()  # Convert to dense array
    for _ in range(max_iter):
        for i in range(0, A.shape[0], K):
            A_block = A_dense[i:i+K, :]
            f_block = f[i:i+K]
            delta_i = np.linalg.solve(A_block.dot(A_block.T), f_block - A_block.dot(x))
            x += np.dot(A_block.T, delta_i)
    return x


def scipy_pseudo_inverse(A, f):
    A_pseudo_inv = np.linalg.pinv(A.toarray())
    x_pi = A_pseudo_inv @ f
    return x_pi

# Load sparse matrix A and ndarray f
A = mmread('poisson2D.mtx')
f = mmread('poisson2D_b.mtx')

# Make sure f is a 1D array
f = np.squeeze(np.array(f))

# Convert A to CSR format for efficient spsolve
A_csr = A.tocsr()

orderings = ['NATURAL', 'MMD_ATA', 'COLAMD']
max_iterations = 1000
threshold = 1e-5

for ordering in orderings:
    print(f"\nSolving with {ordering} ordering:")
    
    # Measure the time
    start_time = time.time()

    if ordering == 'NATURAL':
        # Solve the system using sparse LU factorization with the specified ordering
        lu = splu(A_csr, permc_spec=ordering)
        x = lu.solve(f)
    else:
        # QR factorization
        Q, R = la.qr(A.toarray())  # QR decomposition
        y = np.dot(Q.T, f)  # Let y=Q'.f using matrix multiplication
        x = la.solve(R, y)  # Solve Rx = Q'.f

    end_time = time.time()

    # Print the solution and the time taken for solving
    print("Time taken:", end_time - start_time, "seconds")

    # Compute and print the relative residual
    rel_residual = relative_residual(A_csr, x, f)
    print("Relative Residual:", rel_residual)

    # Check if the solution is accurate enough based on the threshold
    if rel_residual < threshold:
        print("The solution is accurate (below threshold).")
    else:
        print("The solution may not be accurate enough (above threshold).")

    # You can also plot the solution if needed
    plt.plot(x)
    plt.title(f'Solution x with {ordering} ordering')
    plt.show()
    
# 6. Run your block Kaczmarz and block Cimmino algorithms with K=2,4,6 and 8.
K_values = [2, 4, 6, 8]
for K in K_values:
    # Run block Kaczmarz
    start_time = time.time()
    x_block_kaczmarz = block_kaczmarz(A, f, K, max_iter=1000)
    block_kaczmarz_time = time.time() - start_time
    residual_block_kaczmarz = np.linalg.norm(A.dot(x_block_kaczmarz) - f) / np.linalg.norm(f)

    # Print the results for block Kaczmarz
    print(f"Block Kaczmarz (K={K}) Time:", block_kaczmarz_time)
    print(f"Block Kaczmarz (K={K}) Residual:", residual_block_kaczmarz)

    # Run block Cimmino
    start_time = time.time()
    x_block_cimmino = block_cimmino(A, f, K, max_iter=1000)
    block_cimmino_time = time.time() - start_time
    residual_block_cimmino = np.linalg.norm(A.dot(x_block_cimmino) - f) / np.linalg.norm(f)

    # Print the results for block Cimmino
    print(f"Block Cimmino (K={K}) Time:", block_cimmino_time)
    print(f"Block Cimmino (K={K}) Residual:", residual_block_cimmino)
    
# Compare and analyze the results for block Kaczmarz and block Cimmino with Scipy methods
for K in K_values:
    # Results for block Kaczmarz
    x_block_kaczmarz = block_kaczmarz(A, f, K, max_iter=1000)
    residual_block_kaczmarz = np.linalg.norm(A.dot(x_block_kaczmarz) - f) / np.linalg.norm(f)

    # Results for block Cimmino
    x_block_cimmino = block_cimmino(A, f, K, max_iter=1000)
    residual_block_cimmino = np.linalg.norm(A.dot(x_block_cimmino) - f) / np.linalg.norm(f)

    # Results for Scipy pseudo-inverse method
    x_pi = scipy_pseudo_inverse(A, f)
    residual_pi = np.linalg.norm(A.dot(x_pi) - f) / np.linalg.norm(f)

    # Print and compare the results
    print(f"\nResults for K={K}:\n")
    print(f"Block Kaczmarz Residual: {residual_block_kaczmarz}")
    print(f"Block Cimmino Residual: {residual_block_cimmino}")
    print(f"Scipy Pseudo-inverse Residual: {residual_pi}")
    
    
# Step 8: Fit a polynomial for block Cimmino without K=6 and estimate the time result for K=6
K_values_without_6 = [2, 4, 8]  # Exclude K=6
times_without_6 = []
residuals_without_6 = []

for K in K_values_without_6:
    start_time = time.time()
    x_block_cimmino = block_cimmino(A, f, K, max_iter=1000)
    block_cimmino_time = time.time() - start_time
    residual_block_cimmino = np.linalg.norm(A.dot(x_block_cimmino) - f) / np.linalg.norm(f)

    times_without_6.append(block_cimmino_time)
    residuals_without_6.append(residual_block_cimmino)

# Fit a polynomial for the results without K=6
polyfit_results_without_6 = np.polyfit(K_values_without_6, times_without_6, deg=2)
polyfit_function_without_6 = np.poly1d(polyfit_results_without_6)

# Estimate the time result for K=6 using the fitted polynomial
estimated_time_for_K_6_without_6 = polyfit_function_without_6(6)

# Print the results
print("Polynomial coefficients for K values without 6:", polyfit_results_without_6)
print("Estimated time for K=6 without 6:", estimated_time_for_K_6_without_6)


# Step 9: Fit a polynomial for block Cimmino with all K values and extrapolate the time result for K=16
K_values_all = [2, 4, 6, 8]
times_all = []
residuals_all = []

for K in K_values_all:
    start_time = time.time()
    x_block_cimmino = block_cimmino(A, f, K, max_iter=1000)
    block_cimmino_time = time.time() - start_time
    residual_block_cimmino = np.linalg.norm(A.dot(x_block_cimmino) - f) / np.linalg.norm(f)

    times_all.append(block_cimmino_time)
    residuals_all.append(residual_block_cimmino)

# Fit a polynomial for all the results
polyfit_results_all = np.polyfit(K_values_all, times_all, deg=2)
polyfit_function_all = np.poly1d(polyfit_results_all)

# Extrapolate the time result for K=16 using the fitted polynomial
estimated_time_for_K_16_all = polyfit_function_all(16)

# Print the results
print("Polynomial coefficients for all K values:", polyfit_results_all)
print("Extrapolated time for K=16:", estimated_time_for_K_16_all)

# Plot the results for Step 8
plt.figure(figsize=(10, 6))
plt.scatter(K_values_without_6, times_without_6, label='Actual Times (K without 6)', color='blue')
plt.plot(K_values_without_6, polyfit_function_without_6(K_values_without_6), label='Polynomial Fit (K without 6)', color='red')
plt.scatter([6], [estimated_time_for_K_6_without_6], label='Estimated Time for K=6 without 6', color='green', marker='x')
plt.title('Polynomial Fit for Block Cimmino (Without K=6)')
plt.xlabel('K Values')
plt.ylabel('Time (seconds)')
plt.legend()
plt.show()

# Plot the results for Step 9
plt.figure(figsize=(10, 6))
plt.scatter(K_values_all, times_all, label='Actual Times (All K values)', color='blue')
plt.plot(K_values_all, polyfit_function_all(K_values_all), label='Polynomial Fit (All K values)', color='red')
plt.scatter([16], [estimated_time_for_K_16_all], label='Estimated Time for K=16', color='green', marker='x')
plt.title('Polynomial Fit for Block Cimmino (All K values)')
plt.xlabel('K Values')
plt.ylabel('Time (seconds)')
plt.legend()
plt.show()