import argparse
import numpy as np

from generate_dataset import generate_data_sequence as data
from generate_matrix import *
from utils import *

from _network import ReservoirComputingbasedonChaoticBoltzmannMachine as CBM

if __name__ == '__main__':
    MM=2200
    delay=20

    if True:
        T = MM
        #U,D = generate_white_noise(c.delay,T=T+200,)
        U,D = data.generate_white_noise(delay,T=T+200,dist="uniform")
        Up=U[200:]
        Dp=D[200:]
        ### training
        #print("training...")
        max = np.max(np.max(abs(Up)))
        if max>0.5:
            Dp /= max*2
            Up /= max*2

    model = CBM(MM=MM)
    model.generate_network()
    model.fit(train_data=Up,target_data=Dp)

    validation = model.validate(train_data=Up,target_data=Dp)

    Us,Rs,Hx,Hp,Yp = model.show_recode()

    plot1(Up,Us,Rs,Hx,Hp,Yp,Dp,show = 1,save=1,dir_name = "trashfigure",fig_name="fig1")
    plot_MC(Yp,Dp,delay=delay,show = 1,save=1,dir_name = "trashfigure",fig_name="mc1")


