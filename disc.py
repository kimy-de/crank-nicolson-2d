import numpy as np

def tri_disc(N, a):
    
    M0r = np.zeros(N)
    M0r[0] = 1+a
    M0r[1] = -a

    M1r = np.zeros(N)
    M1r[0] = -a
    M1r[1] = 1+2*a
    M1r[2] = -a

    Mlr = np.zeros(N)
    Mlr[-2] = -a
    Mlr[-1] = 1+a

    M = np.zeros((N, N))
    M[0] = M0r
    M[1] = M1r
    M[-1] = Mlr

    for i in range(2,N-1): 
        M[i] = np.roll(M[i-1], 1)

    return M

def conv_mat(N, a, b):
    
    M1r = np.zeros(N*N)
    M1r[1] = a
    M1r[N] = a
    M1r[N+1] = b
    M1r[N+2] = a
    M1r[2*N+1] = a

    M = np.zeros(((N-2)**2, N**2))
    M[0] = M1r
    jump = (N-2)*(np.arange(1,(N-2)**2)-1)+1 
    for i in range(1,(N-2)*(N-2)): 

        if i+1 in jump[1:]:
            M[i] = np.roll(M[i-1], 3) 
        else:
            M[i] = np.roll(M[i-1], 1)

    return M