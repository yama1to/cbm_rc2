
import numpy as np
import matplotlib.pyplot as plt


def generate_narma(N,seed=0,delay=9):
    if type(seed)!=None : np.random.seed(seed=seed)

    """Generate NARMA sequence."""
    N = N + 200
    u = np.random.uniform(0,0.5,(N))

    # Generate NARMA sequence
    d = np.zeros((N))
    
    for i in range(N-1):
        d[i+1] = 0.3*d[i] + 0.05*d[i] * \
            np.sum(d[i-delay:i+1]) + 1.5*u[i-delay]*u[i] + 0.1
    d = d[200:]
    u = u[200:]
    N = N-200
    if np.isfinite(d).all():
        u = u.reshape((N,1))
        d = d.reshape((N,1))
        return u,d
    else:
        print("again")
        return generate_narma(N=N,seed=seed+1)

if __name__ == '__main__':
    u,d = generate_narma(1000)
    plt.plot(u)
    plt.plot(d)
    plt.show()
    print(u.shape,d.shape)