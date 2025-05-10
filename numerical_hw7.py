import numpy as np


A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)
x0 = np.zeros_like(b)
tol = 1e-6
max_iter = 1000
omega = 1.5  # for SOR


x_exact = np.linalg.solve(A, b)

# --- Jacobi ---
def jacobi(A, b, x0, max_iter, tol):
    n = len(b)
    x = x0.copy()
    results = []
    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        results.append(x_new.copy())
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return results

# --- Gauss-Seidel ---
def gauss_seidel(A, b, x0, max_iter, tol):
    n = len(b)
    x = x0.copy()
    results = []
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i+1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        results.append(x.copy())
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            break
    return results

# --- SOR ---
def sor(A, b, x0, max_iter, tol, omega):
    n = len(b)
    x = x0.copy()
    results = []
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i+1, n))
            x[i] = (1 - omega) * x_old[i] + omega * (b[i] - s1 - s2) / A[i][i]
        results.append(x.copy())
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            break
    return results

# --- Conjugate Gradient ---
def conjugate_gradient(A, b, x0, max_iter, tol):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    results = [x.copy()]
    for _ in range(max_iter):
        Ap = A @ p
        alpha = r @ r / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        results.append(x.copy())
        if np.linalg.norm(r_new) < tol:
            break
        beta = r_new @ r_new / (r @ r)
        p = r_new + beta * p
        r = r_new
    return results


def show_result(name, results, x_exact):
    print(f"\n=== {name} Method ===")
    if len(results) > 0:
        print("Iteration 1:", results[0])
    if len(results) > 1:
        print("Iteration 2:", results[1])
    print("Final result :", results[-1])
    print("Exact result :", x_exact)
    print("Max error    :", np.max(np.abs(results[-1] - x_exact)))


jacobi_res = jacobi(A, b, x0, max_iter, tol)
gs_res = gauss_seidel(A, b, x0, max_iter, tol)
sor_res = sor(A, b, x0, max_iter, tol, omega)
cg_res = conjugate_gradient(A, b, x0, max_iter, tol)


show_result("Jacobi", jacobi_res, x_exact)
show_result("Gauss-Seidel", gs_res, x_exact)
show_result("SOR", sor_res, x_exact)
show_result("Conjugate Gradient", cg_res, x_exact)