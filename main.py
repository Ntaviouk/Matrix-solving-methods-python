import numpy as np
import sys


def gaussian(A):
    n = len(A)

    for i in range(n):

        for j in range(i + 1, n):
            ratio = A[j][i] / A[i][i]

            for k in range(n + 1):
                A[j][k] = A[j][k] - ratio * A[i][k]

    x = np.zeros(n)
    x[n - 1] = A[n - 1][n] / A[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = A[i][n]

        for j in range(i + 1, n):
            x[i] = x[i] - A[i][j] * x[j]

        x[i] = x[i] / A[i][i]

    return x


def simple_iteration(A, initial_guess=None, tolerance=0.5 * 10 ** -6, max_iterations=1000):
    A = np.array(A)
    n = A.shape[0]

    if initial_guess is None:
        x = np.zeros(n)
    else:
        x = np.array(initial_guess)

    for _ in range(max_iterations):
        x_new = np.zeros(n)

        for i in range(n):
            sum_val = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (A[i][n] - sum_val) / A[i][i]

        if np.allclose(x, x_new, atol=tolerance):
            return x_new

        x = x_new

    return x


def gauss_seidel(A, tolerance=0.5 * 10 ** -6, max_iterations=1000):
    A = np.array(A)
    n = len(A)
    x = np.zeros(n)
    b = A[:, -1]

    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x_new[j]
            x_new[i] = (b[i] - sigma) / A[i][i]

        if np.allclose(x, x_new, atol=tolerance):
            return x_new

        x = x_new

    return x


def solve_gauss(A):
    return np.linalg.solve(A[:, :-1], A[:, -1])


A = [[3.96, -0.78, -0.35, 2.525],
     [1.18, 3.78, -0.87, 7.301],
     [-0.96, -1.02, 3.68, 9.190]]

B = np.array([[3.96, -0.78, -0.35, 2.525],
              [1.18, 3.78, -0.87, 7.301],
              [-0.96, -1.02, 3.68, 9.190]])

if __name__ == "__main__":
    solution = gaussian(A)
    for i in range(len(solution)):
        print(f"X{i} = {solution[i]}", end='\t')

    print()

    solution = simple_iteration(A)
    for i in range(len(solution)):
        print(f"X{i} = {solution[i]}", end='\t')

    print()

    solution = gauss_seidel(A)
    for i in range(len(solution)):
        print(f"X{i} = {solution[i]}", end='\t')

    print()
    solution = solve_gauss(B)
    print(solution)

