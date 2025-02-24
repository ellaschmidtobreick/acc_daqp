""" 
Generates a multi-parametric quadratic program of the form

minimize_x  0.5 x' H x + (f + F*theta)'x
subject to  A x <= b + B*theta 

"""
import numpy as np

n = 2 # Number of decision variables
m = 5 # Number of constraints
nth = 2 # Number of parameters

def generate_qp(n,m,nth=2):
    # Objective function
    M = np.random.randn(n,n)
    H = M @ M.T # Ensure H symmetric and positive definite. 
    f = np.random.randn(n)
    F = np.random.randn(n,nth)

    # Constraints
    A = np.random.randn(m,n)
    b = np.random.rand(m)
    T = np.random.randn(n,nth)#  A transformation such that x = F0*th is primal feasible
    B = A @ (-T)
    return H,f,F,A,b,B


def generate_rhs(f,F,b,B):
    """ Example QP of the form 

    minimize_x  0.5 x' H x + ftot'x
    subject to  A x <= btot

    for a given theta
    """
    theta = np.random.randn(nth)
    btot = b + B @ theta
    ftot = f + F @ theta
    return ftot,btot
