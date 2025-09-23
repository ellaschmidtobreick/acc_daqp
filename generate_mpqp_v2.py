""" 
Generates a multi-parametric quadratic program of the form

minimize_x  0.5 x' H x + (f + F*theta)'x
subject to  A x <= b + B*theta 

"""
import numpy as np

# n: Number of decision variables
# m: Number of constraints
# nth: Number of parameters

def generate_qp(n,m,given_seed, nth = 2):
    np.random.seed(seed = given_seed)
    # Objective function
    print(n,m,nth)
    M = np.random.randn(n,n)
    H = M @ M.T # Ensure H symmetric and positive definite. 
    f = np.random.randn(n)
    F = np.random.randn(n,nth)

    # Constraints
    A = np.random.randn(m,n)
    b = np.random.rand(m) #rand ensures that b >= 0, which in turns means that the origin is a feasible point (so the problem will have a solution)
    T = np.random.randn(n,nth) # A transformation such that x = T*th is primal feasible
    B = A @ (-T)
    return H,f,F,A,b,B,T

def generate_qp_twosided_constraints(n,m,given_seed, nth = 2):
    np.random.seed(seed = given_seed)
    # Objective function
    M = np.random.randn(n,n)
    H = M @ M.T # Ensure H symmetric and positive definite. 
    f = np.random.randn(n)
    F = np.random.randn(n,nth)

    # Constraints
    m1 = m/2
    A1 = np.random.randn(m1,n)
    A = np.vstack((A1,-A1))
    print(A.shape)
    b1 = np.random.rand(m1) #rand ensures that b >= 0, which in turns means that the origin is a feasible point (so the problem will have a solution)
    b = np.hstack((b1,-b1))     # does not center btot around 0!
    print(b.shape)

    T = np.random.randn(n,nth) # A transformation such that x = T*th is primal feasible
    B = A @ (-T)
    return H,f,F,A,b,B,T

def generate_rhs(f,F,b,B,nth,given_seed):
    """ Example QP of the form 

    minimize_x  0.5 x' H x + ftot'x
    subject to  A x <= btot

    for a given theta
    """
    np.random.seed(seed = given_seed)
    theta = np.random.randn(nth)
    btot = b + B @ theta
    ftot = f + F @ theta
    return ftot,btot

## Try this out
def banded_matrix(n, bandwidth):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-bandwidth), min(n, i+bandwidth+1)):
            M[i, j] = np.random.randn()
    return M

def generate_banded_qp(n, m, given_seed, bandwidth=3, nth=2):
    np.random.seed(given_seed)

    # Banded H
    M = banded_matrix(n, bandwidth)
    H = M @ M.T
    f = np.random.randn(n)
    F = np.random.randn(n, nth)

    # Sparse A
    A = banded_matrix(m, min(bandwidth, n-1))[:, :n]
    b = np.random.rand(m)

    T = np.random.randn(n, nth)
    B = A @ (-T)
    return H, f, F, A, b, B, T

def generate_sparse_qp(n, m, given_seed, density=0.1, nth=2):
    np.random.seed(given_seed)

    # Objective (H)
    M = np.random.randn(n, n)
    H = M @ M.T
    f = np.random.randn(n)
    F = np.random.randn(n, nth)

    # Constraints with sparsity
    mask = np.random.rand(m, n) < density   # keep ~density fraction of entries
    A = np.random.randn(m, n) * mask
    b = np.random.rand(m)

    # Transformation
    T = np.random.randn(n, nth)
    B = A @ (-T)
    return H, f, F, A, b, B, T
