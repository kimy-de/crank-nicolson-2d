import torch
import torch.nn as nn
import numpy as np
import disc
import utils
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("1. Operation Type: ", device)

# grid
N = 100
dt = 1e-4

x = np.linspace(0, 1, N)  # x domain [0,1]
y = np.linspace(0, 1, N)  # y domain [0,1]

h = 1/N
k = dt/(2*h**2)

# initial condition - star shape
u0 = torch.FloatTensor(utils.star_ini(x, y, N)).to(device)

# discretization
A = torch.FloatTensor(disc.tri_disc(N, k)).to(device)
B = torch.FloatTensor(disc.conv_mat(N+2, k, 1-4*k)).to(device)
C = torch.FloatTensor(disc.tri_disc(N, k)).to(device)
print("2. Generated discretization matrices")

# Crank-Nicolson Method
pad = nn.ReplicationPad2d(1)
A_inv = torch.linalg.inv(A)
C_inv = torch.linalg.inv(C)

u_pred = [u0.cpu().numpy().copy()]
u = torch.clone(u0)
u_star = torch.zeros(N,N).to(device)
max_iter = 100

print("3. Started iteration session")
for it in tqdm(range(max_iter-1)):
    
    # step 1
    u_p = pad(u.unsqueeze(0).unsqueeze(0)).reshape(-1,1)
    S = B @ u_p
    S = S.reshape(N,N)
    u_star[1:-1] = S[1:-1]@A_inv.T
     
    u_star[0,0] = k*(u[1,0]-2*u[0,0]+u[0,1])+u[0,0]
    u_star[0,-1] = k*(u[1,-1]-2*u[0,-1]+u[0,-2])+u[0,-1]
    u_star[-1,0] = k*(u[-2,0]-2*u[-1,0]+u[-1,1])+u[-1,0]
    u_star[-1,-1] = k*(u[-2,-1]-2*u[-1,-1]+u[-1,-2])+u[-1,-1]
    u_star[0,1:-1] = k*(-3*u[0,1:-1]+u[1,1:-1]+u[0,:-2]+u[0,2:])+u[0,1:-1]
    u_star[-1,1:-1] = k*(-3*u[-1,1:-1]+u[-2,1:-1]+u[-1,:-2]+u[-1,2:])+u[-1,1:-1]

    # step 2 
    u = C_inv@u_star.T

    u_pred.append(torch.clone(u).cpu().numpy())

utils.resplot(x, y, u_pred, dt, max_iter)
print("4. Completed the code")