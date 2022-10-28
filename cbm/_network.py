# Copyright (c) 2022 Katori lab. All Rights Reserved
import numpy as np
from cbm.generate_matrix import *
from tqdm import tqdm
from cbm.utils import p2s

class ReservoirComputingbasedonChaoticBoltzmannMachine():
    def __init__(self,columns = None,csv = None,id  = None,
                plot = 1,show = False,savefig = False,fig1 = "fig1.png",
                dataset=6,seed:int=0,
                NN=2**8,MM=2200,MM0 = 200,Nu = 1,Nh:int = 100,Ny = 20,
                Temp=1,alpha_i = 0.24,alpha_r = 0.7,alpha_b = 0.,alpha_s = 0.5,
                alpha0=0,alpha1=1,
                beta_i = 0.9,beta_r = 0.2,beta_b = 0.,lambda0 = 0.,do_not_use_tqdm = 0):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = columns # 結果をCSVに保存する際のコラム
        self.csv = csv # 結果を保存するファイル
        self.id  = id
        self.plot = plot # 図の出力のオンオフ
        self.show = show # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = savefig
        self.fig1 = fig1 ### 画像ファイル名

        # config
        self.dataset=dataset
        self.seed:int=seed # 乱数生成のためのシード
        self.NN=NN # １サイクルあたりの時間ステップ
        self.MM=MM # サイクル数
        self.MM0 = MM0 #

        self.Nu = Nu         #size of input
        self.Nh:int = Nh   #815 #size of dynamical reservior
        self.Ny = Ny        #size of output

        self.Temp=Temp
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = alpha_i
        self.alpha_r = alpha_r
        self.alpha_b = alpha_b
        self.alpha_s = alpha_s

        self.alpha0 = alpha0#0.1
        self.alpha1 = alpha1#-5.8

        self.beta_i = beta_i
        self.beta_r = beta_r
        self.beta_b = beta_b

        self.lambda0 = lambda0
        self.do_not_use_tqdm = do_not_use_tqdm

        self.cnt_overflow=None

    
    def generate_network(self,):
        np.random.seed(seed=self.seed)
        self.Wr = generate_random_matrix(self.Nh,self.Nh,self.alpha_r,self.beta_r,distribution="one",normalization="sr",diagnal=0)

        #Wr = bm_weight()
        #Wr = ring_weight()
        #Wr = small_world_weight()
        self.Wb = generate_random_matrix(self.Nh,self.Ny,self.alpha_b,self.beta_b,distribution="one",normalization="none")
        self.Wi = generate_random_matrix(self.Nh,self.Nu,self.alpha_i,self.beta_i,distribution="one",normalization="none")
        self.Wo = np.zeros(self.Nh * self.Ny).reshape((self.Ny, self.Nh))

    def fit(self,train_data,target_data):
        self.run_network(train_data,target_data)
        self.regression(self.Hp,target_data)
        
    def regression(self,reservoir_state,target_data):
        M = reservoir_state[self.MM0:, :]
        G = target_data[self.MM0:, :]

        ### Ridge regression
        if self.lambda0 == 0:
            self.Wo = np.dot(G.T,np.linalg.pinv(M).T)
            #print("a")
        else:
            E = np.identity(self.Nh)
            TMP1 = np.linalg.inv(M.T@M + self.lambda0 * E)
            WoT = TMP1@M.T@G
            self.Wo =WoT.T


    def validate(self,train_data,target_data):
        self.run_network(train_data,target_data)

    def predict(self,train_data,target_data):
        
        return 
        
    def run_network(self,train_data,target_data):
        self.Hp = np.zeros((self.MM, self.Nh))
        self.Yp = np.zeros((self.MM, self.Ny))
        hsign = np.zeros(self.Nh)
        hx = np.zeros(self.Nh)
        #hx = np.random.uniform(0,1,self.Nh) # [0,1]の連続値
        hs = np.zeros(self.Nh) # {0,1}の２値
        hs_prev = np.zeros(self.Nh)
        hc = np.zeros(self.Nh) # ref.clockに対する位相差を求めるためのカウント
        hp = np.zeros(self.Nh) # [-1,1]の連続値
        ht = np.zeros(self.Nh) # {0,1}
        
        yp = np.zeros(self.Ny)
        ys = np.zeros(self.Ny)
        yx = np.zeros(self.Ny)
        if self.plot:
            self.Hx = np.zeros((self.MM*self.NN, self.Nh))
            self.Hs = np.zeros((self.MM*self.NN, self.Nh))
            self.Yx = np.zeros((self.MM*self.NN, self.Ny))
            self.Ys = np.zeros((self.MM*self.NN, self.Ny))
            #ysign = np.zeros(Ny)
            #yc = np.zeros(Ny)
            self.Us = np.zeros((self.MM*self.NN, self.Nu))
            self.Ds = np.zeros((self.MM*self.NN, self.Ny))
            self.Rs = np.zeros((self.MM*self.NN, 1))

        rs = 1
        any_hs_change = True
        count =0
        m = 0

        tmp = np.zeros((self.Nh,self.NN))
        hp = np.zeros((self.Nh))


        for n in tqdm(range(self.NN * self.MM),disable=self.do_not_use_tqdm):
            theta = np.mod(n/self.NN,1) # (0,1)
            rs_prev = rs
            hs_prev = hs.copy()

            rs = p2s(theta,0)# 参照クロック
            us = p2s(theta,train_data[m]) # エンコードされた入力
            #us = p2s(theta,Wi@(2*Up[m]-1))
            ds = p2s(theta,target_data[m]) #
            ys = p2s(theta,yp)
            #print(us == Wi@(2*p2s(theta,Up[m])-1))
            # print("aaaaaaaaaaaaaaaaaaa")
            # print(us)
            # print(Wi@(2*p2s(theta,Up[m])-1))
            sum = np.zeros(self.Nh)
            #sum += self.alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
            sum += self.alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
            sum += self.Wi@(2*us-1) # 外部入力
            #sum += us
            #Wr = generate_random_matrix(self.Nh,self.Nh,self.alpha_r,self.beta_r,distribution="one",normalization="sr",diagnal=0)
            sum += self.Wr@(2*hs-1) # リカレント結合
            #print(tmp.shape)
            # sum += self.Wr@(2*p2s(theta,hp)-1)
            #sum+=  Wr@(2*p2s(theta,hp2)-1)/2
            #sum += Wr@(2*tmp[:,int(n%self.NN)]-1) # リカレント結合

            # if mode == 0:
            #    sum += Wb@(2*ys-1)
            # if mode == 1:  # teacher forcing
            #    sum += Wb@(2*ds-1)


            
            hsign = 1 - 2*hs
            hx = hx + hsign*(1.0+np.exp(hsign*sum/self.Temp))*self.dt
            hs = np.heaviside(hx+hs-1,0)
            hx = np.fmin(np.fmax(hx,0),1)
            
            #tmp[:,int(n%self.NN)] = hs

            hc[(hs_prev == 1)& (hs==0)] = count
            
            # hc[(hs_prev == 0)& (hs==1)] =-n
            # hc[(hs_prev == 1)& (hs==0)] +=n
            # ref.clockの立ち上がり
            if rs_prev==0 and rs==1:
                hp = 2*hc/self.NN-1 # デコード、カウンタの値を連続値に変換

                hc = np.zeros(self.Nh) #カウンタをリセット
                ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト
                yp = self.Wo@hp
                # record    
                self.Hp[m]=hp
                self.Yp[m]=yp
                count = 0
                m += 1

            #境界条件
            if n == (self.NN * self.MM-1):
                hp = 2*hc/self.NN-1 # デコード、カウンタの値を連続値に変換
                yp = self.Wo@hp
                # record
                self.Hp[m]=hp
                self.Yp[m]=yp
            
            
            count += 1
            any_hs_change = np.any(hs!=hs_prev)
            #Hx[n]=hx
            if self.plot:
            # record
                self.Rs[n]=rs
                self.Hx[n]=hx
                self.Hs[n]=hs
                self.Yx[n]=yx
                self.Ys[n]=ys
                self.Us[n]=us
                self.Ds[n]=ds

        # オーバーフローを検出する。
        self.cnt_overflow=0
        for m in range(2,self.MM-1):
            tmp = np.sum( np.heaviside( np.fabs(self.Hp[m+1]-self.Hp[m]) - 0.6 ,0))
            self.cnt_overflow += tmp

    def show_recode(self,):
        if not self.plot:
            return {},{},{},self.Hp[self.MM0:],self.Yp[self.MM0:]

        else:
            return self.Us,self.Rs,self.Hx,self.Hp[self.MM0:],self.Yp[self.MM0:]

    