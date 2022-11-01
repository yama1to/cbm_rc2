from generate_datasets.generate_data_sequence_ou import datasets
import numpy as np
import matplotlib.pyplot as plt

def dataset(delay,T):
    U,D = datasets(T=T+200,delay_s=delay)
    Up=U[200:]
    Dp=D[200:]
    ### training
    #print("training...")
    max = np.max(np.max(abs(Up)))
    if max>0.5:
        Dp /= max*2
        Up /= max*2

    return Up,Dp


def evaluate(Yp,Dp,delay):
    DC = np.zeros(delay)
    for k in range(delay):
        corr = np.corrcoef(np.vstack((Dp.T[k, k:], Yp.T[k, k:])))   #相関係数
        DC[k] = corr[0, 1] ** 2    #決定係数 = 相関係数 **2
    MC = np.sum(DC)
    return DC,MC

def plot(Yp,Dp,delay=20,show = 1,save=1,dir_name = "trashfigure",fig_name="mc1"):               
    DC,MC = evaluate(Yp,Dp,delay)
    plt.plot(DC)
    plt.ylabel("determinant coefficient")
    plt.xlabel("Delay k")
    plt.ylim([0,1])
    plt.xlim([0,delay])
    plt.title('MC ~ %3.2lf' % MC, x=0.8, y=0.7)
    if save:plt.savefig("./{}/{}".format(dir_name,fig_name))
    if show :plt.show()