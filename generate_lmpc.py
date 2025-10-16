import numpy
from lmpc import MPC,ExplicitMPC

# import os, julia
# os.add_dll_directory(r"acc_daqp_env/julia_env/pyjuliapkg/install/bin")

# Continuous time system dx = A x + B u

# system matrix: 4×4 for 4 states: cart position, cart velocity, pendulum angle, pendulum angular velocity) (per row)
# x^dot is a column vector with [x_cart, v_cart, theta, theta^dot]^T
# the rows determine which variable gets influenced by which other variable
# the columns determine which other variables each variable influences
# position and velocity (row 1+2) should be between +-10 to get more activations
A = numpy.array([[0, 1, 0, 0], [0, -10, 9.81, 0], [0, 0, 0, 1], [0, -20, 39.24, 0]])
# input matrix (4×1, scaled by 100 here)
B = 100*numpy.array([0,1.0,0,2.0])
# output matrix (maps states to outputs, here measuring cart position and pendulum angle)
C = numpy.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])

# These matrices define F, G, C in the discrete-time QP formula after discretization:
# x_k+1 = Fx_k+Gu_k,    y_k = Cx_k


# create an MPC control with sample time 0.01, prediction horizon 10 and control horizon 5 
# prediction horizon Np: N in the formula (number of future steps considered)
# Nc: control horizon (number of free inputs to optimize; remaining inputs may be held constant)
# Data Daniel: Np = 50, Nc = 5,10, 25, 50

# could be scenarios with varying horizons (3 differnet, tradeoff performance vs. computational time)
# given a MPC problem with fixed horizon has a fixed sparsity pattern
Np,Nc = 50,50
# Ts: sample time (discrete-time conversion of A, B to F, G)
Ts = 0.01
# creates the QP problem template with constraints
mpc = MPC(A,B,Ts,C=C,Nc=Nc,Np=Np);

# set the objective functions weights
# Q: weight on output y=Cx for each measured output (cart position and pendulum angle)
# R: weight on input magnitude uTRu (here zero - input magnitude is not penalized)
# Rr: weight on input change delta u(k) = u(k)-u(k-1)
# directly used to built matrices H and f (long complicated formula to explicitly built it)
# scale down R<1 to get more active constraints
mpc.set_objective(Q=[1.44,1],R=[0.0],Rr=[0.001])


# add more constraints
mpc.add_constraint(Ax=numpy.array([[1,0,0,0], [0,0,1,0]]), lb=[-10,-45*numpy.pi/180], ub=[10,45*numpy.pi/180]) # position and angle constraints (position between -10 and 10, angle between -45 and 45 degrees)
# even denser: also constraints on velocity(state 2) and angular velocity (state 4)


# set actuator limits
mpc.set_bounds(umin=[-2.0],umax=[2.0])

# run 
res = mpc.mpqp(singlesided=True)

H = res["H"]
f = res["f"]
H_theta = res["H_theta"]
f_theta = res["f_theta"]
A = res["A"]
b = res["b"]
W = res["W"]
senses = res["senses"]
prio = res["prio"]
has_binaries = res["has_binaries"]

print("H",H)
print("condition number",numpy.linalg.cond(H))
print("f",f)
print("H_theta",H_theta)
print("f_theta",f_theta)
print("A",A)
print("b",b)
print("W",W)
print("senses",senses)
print("prio",prio)
print("has_binaries",has_binaries)

# save data
numpy.savez(f"data/mpc_mpqp_N{int(A.shape[1])}_R_0001.npz", H=H, f=f, f_theta=f_theta, A=A, b=b, W=W)



# data = np.load('data/mpc_mpqp_N5.npz')

# # ['H', 'f', 'f_theta', 'A', 'b', 'W']
# print(data.files)

# print(data['H'])    # 5x5
# print(data['f'])    # 5x1 (all 0)
# print(data['f_theta'])  # 5x7
# print(data['A'])    # (5+5)x5 (upper & lower constraints)
# print(data['b'])    # (5+5))x1 (all 2)
# print(data['W'])    # (5+5)x7 (all 0)

# data = np.load('data/mpc_mpqp_N10.npz')
# print(data.files)

# print(data['H'])    # 10x10
# print(data['f'])    # 10x1 (all 0)
# print(data['f_theta'])  # 10x7
# print(data['A'])    # (10+10)x10 (upper & lower constraints)
# print(data['b'])    # (10+10)x1 (all 2)
# print(data['W'])    # (10+10)x7 (all 0)


# data = np.load('data/mpc_mpqp_N25.npz')
# print(data.files)

# data = np.load('data/mpc_mpqp_N50.npz')
# print(data.files)


