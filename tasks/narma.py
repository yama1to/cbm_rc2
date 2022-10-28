from generate_datasets.generate_data_sequence_narma import generate_narma
import numpy as np
import matplotlib.pyplot as plt

def dataset(MM1,MM2,delay=9,seed1=0,seed2=1000):

    U1,D1= generate_narma(MM1,delay,seed1)
    U2,D2= generate_narma(MM2,delay,seed2)

    return U1,D1,U2,D2



def evaluate(Yp,Dp,):
    def calc(Yp,Dp):
        error = (Yp-Dp)**2
        NMSE = np.mean(error)/np.var(Dp)
        RMSE = np.sqrt(np.mean(error))
        NRMSE = RMSE/np.var(Dp)
        return RMSE,NRMSE,NMSE

    Yp = np.tanh(Yp)
    Dp = np.tanh(Dp)
    RMSE,NRMSE,NMSE = calc(Yp,Dp)
    #print(1/np.var(Dp))
    print('RMSE ={:.3g}'.format(RMSE))
    print('NMSE ={:.3g}'.format(NMSE))
    print('NRMSE ={:.3g}'.format(NRMSE))
    return RMSE,NMSE,NRMSE


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
