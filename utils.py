import numpy as np
import matplotlib.pyplot as plt


def star_ini(x, y, N):
    R0 = .25
    eps = 5 * 0.01 / (2 * np.sqrt(2) * np.arctanh(0.9))
    pn = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if x[i] > 0.5:
                theta = np.arctan2(y[j] - 0.5, x[i] - 0.5)
                pn[i, j] = np.tanh(
                    (R0 + 0.1 * np.cos(6 * theta) - (np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2))) / (
                                np.sqrt(2.0) * eps))
            else:
                theta = np.pi + np.arctan2(y[j] - 0.5, x[i] - 0.5)
                pn[i, j] = np.tanh(
                    (R0 + 0.1 * np.cos(6 * theta) - (np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2))) / (
                                np.sqrt(2.0) * eps))
                
    return pn

def resplot(x, y, u_pred, dt, max_iter):
    fig = plt.figure(figsize=(8, 2))
    plt.subplot(141)
    plt.imshow(u_pred[0], interpolation='nearest', cmap='jet',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$u_0$', fontsize=15)
    
    plt.subplot(142)
    l = int(0.25*max_iter)
    plt.imshow(u_pred[l], interpolation='nearest', cmap='jet',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$u_{%d}$ (t=%.3f)' %(l, dt*l), fontsize=15)
    
    plt.subplot(143)
    l = int(0.5*max_iter)
    plt.imshow(u_pred[l], interpolation='nearest', cmap='jet',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$u_{%d}$ (t=%.3f)' %(l, dt*l), fontsize=15)
    
    plt.subplot(144)
    plt.imshow(u_pred[-1], interpolation='nearest', cmap='jet',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.axis('off')
    plt.title('$u_{%d}$ (t=%.3f)' %(max_iter, dt*max_iter), fontsize=15)
    
    plt.savefig('./cn2d.png')