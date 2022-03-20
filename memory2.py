# Copyright (c) 2022 Katori lab. All Rights Reserved

import argparse
import numpy as np
import sys
import os

from generate_datasets.generate_data_sequence import generate_white_noise as data
from utils import calc_MC, plot1,plot_MC
from explorer import common
from _network import ReservoirComputingbasedonChaoticBoltzmannMachine as CBM

class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 0 # 図の出力のオンオフ
        self.show = False # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = False
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=6
        self.seed:int=1 # 乱数生成のためのシード
        self.NN=2**2 # １サイクルあたりの時間ステップ
        self.MM=2200 # サイクル数
        self.MM0 = 200 #

        self.Nu = 1         #size of input
        self.Nh:int = 100   #815 #size of dynamical reservior
        self.Ny = 20        #size of output

        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.24
        self.alpha_r = 0.9
        self.alpha_b = 0.
        self.alpha_s = 0.5

        self.alpha0 = 0#0.1
        self.alpha1 = 1#-5.8

        self.beta_i = 0.9
        self.beta_r = 0.02
        self.beta_b = 0.

        self.lambda0 = 0.

        self.delay = 20

        # ResultsX
        self.RMSE1=None
        self.RMSE2=None
        self.MC = None
        self.MC1 = None 
        self.MC2 = None
        self.MC3 = None
        self.MC4 = None
        self.cnt_overflow=None

def execute(c):
    np.random.seed(c.seed)
    if True:
        T = c.MM
        #U,D = generate_white_noise(c.delay,T=T+200,)
        U,D = data(c.delay,T=T+200,dist="uniform")
        Up=U[200:]
        Dp=D[200:]
        ### training
        #print("training...")
        max = np.max(np.max(abs(Up)))
        if max>0.5:
            Dp /= max*2
            Up /= max*2

    model = CBM(columns = c.columns,csv = c.csv,id  = c.id,
                plot = c.plot,show = c.show,savefig = c.savefig,fig1 = c.fig1,
                dataset=c.dataset,seed=c.seed,
                NN=c.NN,MM=c.MM,MM0 = c.MM0,Nu = c.Nu,Nh= c.Nh,Ny = c.Ny,Temp=c.Temp,
                alpha_i = c.alpha_i,alpha_r = c.alpha_r,alpha_b = c.alpha_b,alpha_s = c.alpha_s,
                beta_i = c.beta_i,beta_r = c.beta_r,beta_b = c.beta_b,
                lambda0 = c.lambda0,delay = c.delay)

    model.generate_network()
    model.fit(train_data=Up,target_data=Dp)

    validation = model.validate(train_data=Up,target_data=Dp)

    _,MC = calc_MC(model.Yp,Dp,model.delay)
    print("MC={:.2f}".format(MC))

    if c.plot:
        Us,Rs,Hx,Hp,Yp = model.show_recode()
        plot1(Up,Us,Rs,Hx,Hp,Yp,Dp,show = c.plot,save=1,dir_name = "trashfigure",fig_name="fig1")
        plot_MC(Yp,Dp,delay=c.delay,show = c.plot,save=1,dir_name = "trashfigure",fig_name="mc1")



if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()
    
    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
