from generate_datasets.generate_data_sequence import generate_xor
import numpy as np
import matplotlib.pyplot as plt

def dataset(MM1,MM2,tau=2):
    U,D = generate_xor(MM1+MM2 +tau,tau)
    U = U[:MM1]
    D = D[:MM1]
    Up = np.tanh(U)

    return Up,D



def evaluate(Yp,Dp,MM,tau=2):
    # 評価（ビット誤り率, BER）
    train_Y_binary = np.zeros(MM-tau)

    train_Y = Yp[tau:]        #(T-tau,1)
    Dp      = Dp[tau:]          #(T-tau,1)

    #閾値を0.5としてバイナリ変換する
    for n in range(MM-tau):
        train_Y_binary[n] = np.heaviside(train_Y[n]-np.tanh(0.5),0)
    
    BER = np.linalg.norm(train_Y_binary-Dp[:,0], 1)/(MM-tau)

    print('BER ={:.3g}'.format(BER))

    return BER


def plot(Up,Hp,Yp,Dp,show,save,dir_name = "trashfigure",fig_name="mc1"):
    fig=plt.figure(figsize=(20, 12))
    Nr=4
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    #ax.set_title("input")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    #ax.set_title("decoded reservoir states")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    #ax.set_title("predictive output")
    #ax.plot(train_Y)
    ax.plot(Yp)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    #ax.set_title("desired output")
    ax.plot(Dp)
    
    if save:plt.savefig("./{}/{}".format(dir_name,fig_name))
    if show :plt.show()
