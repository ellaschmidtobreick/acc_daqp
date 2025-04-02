import numpy as np
import daqp
import ipdb
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
    print("run: fix component")
    print("B in fixed component",B)
    print("lam",lam)
    print("p",p)
    print("-lam[B]/p[B]",-lam[B]/p[B])
    j = np.argmin(-lam[B]/p[B])
    print("j",j)
    W_new = [int(i) for i in W if int(i)!=int(B[j])]
    print("W",W)
    print("new W", W_new)
    print("lam",lam)
    lam = lam - (lam[B[j]]/p[B[j]])*p
    lam[np.abs(lam) < 1e-12] = 0
    print("new lam",lam)
    lam_star = np.copy(lam)
    print("new lam_star",lam_star)
    return lam,lam_star,mu, W_new           


def daqp_self(H,f,A,b,sense,W):
    # transform matrices into M,d,v,R
    try:
        R = np.linalg.cholesky(H, upper = True)
    except:
        print("Invalid input")
        raise ValueError("Invalid input")
    M = A @ np.linalg.inv(R)
    print("shape of M", M.shape)
    row_norm = np.linalg.norm(M, axis=1, keepdims=True, ord=2)
    print("row_norm",row_norm)
    print("M before", M)
    M_normed = M / row_norm  # Normalize M
    print("M after",M_normed)
    v = np.linalg.inv(R).T @f
    d = b.reshape(-1) + M @ v
    print("d before", d)
    d_normed = d / row_norm.reshape(-1)
    print("d after",d_normed)
    M = np.copy(M_normed)
    d = np.copy(d_normed)
    print("initial values")
    print("R",R)
    print("M", M)
    print("v", v)
    print("d",d)
    print("---------")
    # initial values
    lam = np.array([0 for i in range(len(b))],dtype=c_double)
    mu = np.array([0 for i in range(len(b))],dtype=c_double)
    lam_star =  np.array([0 for i in range(len(b))],dtype=c_double)
    print("initial lambda", lam)
    print("initial working set", W)
    print("mu",mu)
    iter = 0
    while True:
        if iter == 100:
            return None,None,W,iter
        iter += 1
        print("---------")
        print("Iteration",iter)
        W = np.sort(W)
        W = list(W)
        print("W",W, "len(W)", len(W))
        W_bar =  np.sort([x for x in range(len(lam)) if x not in W])
        print("W_bar", W_bar)
        mu = np.array([0 for i in range(len(b))],dtype=c_double)
        print("mu at the beginning of the round",mu)
        p = np.array([0 for i in range(len(b))],dtype=c_double)
        if len(W)==0 or is_invertible(M[W,:]@ M[W,:].T): # step 2
            print("case: MkMkT invertible")
            print("lam_star",lam_star)
            print("lam",lam)
            print("len(W)",len(W))
            if len(W)>0:
                print("M[W]",M[W,:])
                print("M[W] @M[W].T",M[W,:] @M[W,:].T)
                print("np.linalg.inv(M[W] @M[W].T)",np.linalg.inv(M[W] @M[W].T))
                print("d[W]",d[W],d[W].shape)
                print("- np.linalg.inv(M[W] @M[W].T) @d[W]",- np.linalg.inv(M[W] @M[W].T) @d[W])
                lam_star[W] = (- np.linalg.inv(M[W,:] @M[W,:].T) @d[W]) #.reshape(-1) # solve the system, step 3
            else:
                lam_star=np.array([0 for i in range(len(b))],dtype=c_double)
            print("lam",lam)
            print("lam_star",lam_star)
            if (np.array(lam_star) >= -1e-12).all(): # step 4
                print("case: all lam_star>=0")
                print("len(W), len(W_bar)",len(W), len(W_bar))
                print("mu before update", mu)
                if len(W)>0:
                    if len(W_bar)>0: # constraints are split between W and W_bar
                        print("M[W_bar]",M[W_bar,:])
                        print("M[W]",M[W,:])
                        print("lam_star[W]",lam_star[W])
                        print("d[W_bar]",d[W_bar])
                        print(" M[W_bar] @M[W].T", M[W_bar,:] @M[W,:].T)
                        print("M[W_bar] @M[W].T@lam_star[W]",M[W_bar,:] @M[W,:].T@lam_star[W])
                        print("M[W_bar] @M[W].T@lam_star[W]+d[W_bar]",M[W_bar,:] @M[W,:].T@lam_star[W]+d[W_bar])
                        mu[W_bar] = M[W_bar,:] @M[W,:].T@lam_star[W]+d[W_bar]
                    else: # all constraints in W
                        print("case: all constraints in W_bar")
                        mu = np.array([0 for i in range(len(b))],dtype=c_double)
                elif len(W)==0: # all constraints in W_bar
                    mu = np.copy(d)
                print("mu after update",mu)
                print("update lam now")
                print("mu",mu)
                lam = np.copy(lam_star)
                print("lam",lam)
                print("mu",mu)
                if (np.array(mu) >= -1e-6).all(): # step 6
                    # optimum found
                    print("case: all mu >=0")
                    lam_star_normalized = np.divide(lam_star, row_norm.reshape(-1), out=np.zeros_like(lam_star), where=row_norm.reshape(-1)!=0)
                    print("print all optimal values",R,M,lam_star_normalized, W,iter)
                    # problem: algorithm doesn't stop, when working set contains everything
                    print(lam_star.dtype)
                    print((lam_star/row_norm.reshape(-1)).dtype)
                    return -np.linalg.inv(R)@(M[W,:].T@lam_star[W]+v), lam_star_normalized, W, iter
                else: # step 7                  
                    print("case: not all mu >=0")
                    print("mu_W_bar",mu[W_bar])
                    j = np.argmin(mu[W_bar])
                    print("j",W_bar[j])
                    W.append(int(W_bar[j]))
                    print("new W for next round",W)
                    mu[j]=0     # was not there in the old version
            else:             
                print("case: lam_star<0") # step 8
                print("lam_star",lam_star)
                print("lam",lam)
                p = lam_star-lam # step 9
                print("p",p)
                B = [i for i in W if lam_star[i]<1e-12]
                print("B",B)
                lam,lam_star,mu, W = fix_component(lam,lam_star,mu, W, B, p) # step 10
                print("lm_star",lam_star)
        else:
            print("MkMkT not invertible")
            print("M[W]@ M[W].T",M[W]@ M[W].T)
            print("rank",np.linalg.matrix_rank(M[W,:]@ M[W,:].T))
            print("W",W)
            print("M[W]",M[W,:])
            MM=Matrix(M[W,:] @ M[W,:].T) # step 12
            MM = np.array(MM, dtype=np.float64)
            print("MM",MM)
            nullspace = scipy.linalg.null_space(MM)
            print("nullsapce",nullspace)
            print("length of MM nullspace",len(nullspace))
            if len(nullspace)==0:
                p[W] = np.zeros((1,len(W)))
            else:
                if nullspace.shape[1]>1:
                    p[W] = nullspace[0]
                    print("p",p.T)
                    print("d",d)
                    print("multiplication",p.T@d)
                    if p.T@d>= 1e-12:
                        p = -p

                else:
                    p[W] = nullspace.reshape(-1)
                    print("p",p.T)
                    print("d",d)
                    print("multiplication",p.T@d)
                    if p.T@d>= 1e-12:
                        p = -p
                    
            print("p",p)          
            B = [i for i in W if p[i]<1e-12] #step 13
            lam,lam_star,mu, W =  fix_component(lam,lam_star,mu,W,B,p) # step 14
            print("lam_star",lam_star)


           
        

# H = np.array([[1, 0], [0, 1]],dtype=c_double)
# f = np.array([2, 2],dtype=c_double)
# A = np.array([[1, 0], [0, 1]],dtype=c_double)
# bupper = np.array([1,1],dtype=c_double)
# b = np.array([1,1],dtype=c_double)
# blower= np.array([-5,-5],dtype=c_double)
# sense = np.array([0,0],dtype=c_int)
# # intial working set
# W = [] 
# x_star, lam_star, W, iter = daqp_self(H,f,A,b,sense,W)
# #print(f"x*={x_star}, lam*={lam_star},W={W}, #iterations: {iter}")

# x,fval,exitflag,info = daqp.solve(H,f,A,bupper,blower,sense)
# #print("Optimal solution:")
# #print(x)
# #print("Exit flag:",exitflag)
# #print("Info:",info)

