import numpy as np
import daqp
import ipdb
import self_implement_daqp_archive
import numpy as np
from ctypes import * 
import ctypes.util
from sympy import Matrix
import sys
import scipy
from sklearn.preprocessing import normalize

def is_invertible(A):
    return A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]

def fix_component(lam,lam_star,mu, W, B, p):
    j = np.argmin(-lam[B]/p[B])
    W_new = [int(i) for i in W if int(i)!=int(B[j])]
    lam = lam - (lam[B[j]]/p[B[j]])*p
    lam[np.abs(lam) < 1e-12] = 0
    lam_star = np.copy(lam)
    return lam,lam_star,mu, W_new           


def daqp_self(H,f,A,b,sense,W):
    # transform matrices into M,d,v,R
    try:
        R = np.linalg.cholesky(H, upper = True)
    except:
        raise ValueError("Invalid input")
    M = A @ np.linalg.inv(R)
    row_norm = np.linalg.norm(M, axis=1, keepdims=True, ord=2)
    M_normed = M / row_norm  # Normalize M
    v = np.linalg.inv(R).T @f
    d = b.reshape(-1) + M @ v
    d_normed = d / row_norm.reshape(-1)
    M = np.copy(M_normed)
    d = np.copy(d_normed)
    
    # initial values
    lam = np.array([0 for i in range(len(b))],dtype=c_double)
    mu = np.array([0 for i in range(len(b))],dtype=c_double)
    lam_star =  np.array([0 for i in range(len(b))],dtype=c_double)

    iter = 0
    while True:
        if iter == 10:
            break
        iter += 1
        W = np.sort(W)
        W = list(W)
        W_bar =  np.sort([x for x in range(len(lam)) if x not in W])
        mu = np.array([0 for i in range(len(b))],dtype=c_double)
        p = np.array([0 for i in range(len(b))],dtype=c_double)
        if len(W)==0 or is_invertible(M[W,:]@ M[W,:].T): # step 2
            if len(W)>0:
                lam_star[W] = (- np.linalg.inv(M[W,:] @M[W,:].T) @d[W]) #.reshape(-1) # solve the system, step 3
            else:
                lam_star=np.array([0 for i in range(len(b))],dtype=c_double)
            if (np.array(lam_star) >= -1e-12).all(): # step 4
                if len(W)>0:
                    if len(W_bar)>0: # constraints are split between W and W_bar
                        mu[W_bar] = M[W_bar,:] @M[W,:].T@lam_star[W]+d[W_bar]
                    else: # all constraints in W
                        mu = np.array([0 for i in range(len(b))],dtype=c_double)
                elif len(W)==0: # all constraints in W_bar
                    mu = np.copy(d)
                lam = np.copy(lam_star)
                if (np.array(mu) >= -1e-6).all(): # step 6
                    # optimum found
                    lam_star_normalized = np.divide(lam_star, row_norm.reshape(-1), out=np.zeros_like(lam_star), where=row_norm.reshape(-1)!=0)
                    return -np.linalg.inv(R)@(M[W,:].T@lam_star[W]+v), lam_star_normalized, W, iter
                else: # step 7                  
                    j = np.argmin(mu[W_bar])
                    W.append(int(W_bar[j]))
                    mu[j]=0     
            else:             
                p = lam_star-lam # step 9
                B = [i for i in W if lam_star[i]<1e-12]
                lam,lam_star,mu, W = fix_component(lam,lam_star,mu, W, B, p) # step 10
        else:
            MM=Matrix(M[W,:] @ M[W,:].T) # step 12
            MM = np.array(MM, dtype=np.float64)
            nullspace = scipy.linalg.null_space(MM)
            if len(nullspace)==0:
                p[W] = np.zeros((1,len(W)))
            else:
                if nullspace.shape[1]>1:
                    p[W] = nullspace[0]
                    if p.T@d>= 1e-12:
                        p = -p

                else:
                    p[W] = nullspace.reshape(-1)

                    if p.T@d>= 1e-12:
                        p = -p
   
            B = [i for i in W if p[i]<1e-12] #step 13
            lam,lam_star,mu, W =  fix_component(lam,lam_star,mu,W,B,p) # step 14
