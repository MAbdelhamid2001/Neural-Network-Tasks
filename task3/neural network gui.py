#!/usr/bin/env python
# coding: utf-8

# In[67]:


import tkinter as tk
from tkinter import messagebox,ttk
from dynamic_penguin_Lastest_update import * ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder ,MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf




def exe_GUI():

    ###
#     data=pd.read_csv(r'F:\mohamed\4th year new\NN\labs\Lab3 (1)\penguins.csv')
#     data=preprocessing(data)

    ###
    top=tk.Tk()
    top.geometry('500x500')
    top.minsize(500, 500) 
    top.maxsize(500, 500)
    top.title('Neural Network GUI')
    top['background']='#856ff8'
#####################
    ###activation
    act_l=tk.Label(top,text="Active_fun",width=20,height=2)
    act_l.place(x=30,y=0)

    act_var=tk.StringVar()
    act=ttk.Combobox(top,textvariable=act_var,width=20)
    act['values']=('Sigmoid','TanH')
    act.current(0)
    act.place(x=180,y=0)
    
    def retrieve_act():
        act_v=act.get()
        print(act_v)
        return(act_v)
    
    Button = tk.Button(top, text = "Submit", command = retrieve_act,width=8,height=1)
    Button.place(x=330,y=0)
    
    
######################
        # #entr # hidden layers 
    hl=tk.Label(top,text="Number of Hidden layers",width=20,height=2)
    hl.place(x=30,y=50)
    hl_entry=tk.Entry(top,width=20)
    hl_entry.place(x=180,y=50)
    #lr_entry.focus_set()
    hl_entry.insert(0,5)
    def callback_hl():
        hl_v=hl_entry.get()
        print(hl_v)
        return hl_v
    
    B=tk.Button(top,text='insert',command=callback_hl,width=8,height=1)
    B.place(x=330,y=50)


    ##enter neurons in each layer
    neo=tk.Label(top,text="Number neurons in layers",width=20,height=2)
    neo.place(x=30,y=100)
    neo_entry=tk.Entry(top,width=20)
    neo_entry.place(x=180,y=100)
    neo_entry.insert(0,'5,6,7,8,9')

    #ep_entry.focus_set()
    def callback_neo():
        neo_v=neo_entry.get()
        print(neo_v)
        return neo_v

    B=tk.Button(top,text='insert',command=callback_neo,width=8,height=1)
    B.place(x=330,y=100)
    
    
######################    

###################
    # #entr Learning Rate
    lr=tk.Label(top,text="LearnRate",width=20,height=2)
    lr.place(x=30,y=150)
    lr_entry=tk.Entry(top,width=20)
    lr_entry.place(x=180,y=150)
    #lr_entry.focus_set()
    lr_entry.insert(0,0.05)
    def callback_lr():
        lr_v=lr_entry.get()
        print(lr_v)
        return lr_v
    
    B=tk.Button(top,text='insert',command=callback_lr,width=8,height=1)
    B.place(x=330,y=150)

########
    ##enter epochs
    ep=tk.Label(top,text="Epochs",width=20,height=2)
    ep.place(x=30,y=200)
    ep_entry=tk.Entry(top,width=20)
    ep_entry.place(x=180,y=200)
    ep_entry.insert(0,20)

    #ep_entry.focus_set()
    def callback_ep():
        ep_v=ep_entry.get()
        print(ep_v)
        return ep_v

    B=tk.Button(top,text='insert',command=callback_ep,width=8,height=1)
    B.place(x=330,y=200)
############

    ##Addbias
    bias_l=tk.Label(top,text="Bias",width=20,height=2)
    bias_l.place(x=30,y=250)

    def add_bias():
        b_v=check_bias.get()
        print(f'bias state{b_v}')
        return b_v
    
    check_bias=tk.IntVar()
    c1=tk.Checkbutton(top,variable=check_bias,onvalue=1,offvalue=0,command=add_bias)
    c1.place(x=180,y=250)
    check_bias.get()
#########


    ##Run
    def callback_Run():
        msg=messagebox.showinfo('welcome','Code is Running')
        print("Code is Running")

        Output.delete('1.0', tk.END)
        ######
        hl_v=int(callback_hl())#number of hidden layer
        LR=float(callback_lr())
        epochs=int(callback_ep())
        Add_bias=add_bias()
        activ_func=retrieve_act()
        
        if activ_func =='Sigmoid':#Sigmoid','TanH'
            activ_func='sigmoid' ##to be created
        else:
            activ_func='tanh'##to be created
            
        neo_v= callback_neo()#number of neorons in each hidded layer
        
        
        neo_v=neo_v.split(',')
        neo_v=[int(i) for i in neo_v]
        print("-----Enters-----")
        print(f"number of Hidden layers ={hl_v}")
        print(f"neorons in each layer ={neo_v}")
        print(f"Activation function ={activ_func}")
        print(f"Number of Epochs ={epochs}")
        print(f"Learning Rate ={LR}")
        print(f"Add Bias  ={Add_bias}")
        
        
        ######

        data = pd.read_csv(r'F:\mohamed\4th_year_new\NN\labs\Lab3 (1)\penguins.csv')
        x,xt,y,yt=preprocessing_penguins(data)
        
        
        #w_ ,b_=perceptron_algorithm(x_train,y_train,Add_bias,LR,epochs,activ_func,thresh=thres)
        
        activ=activ_func
        addbias=Add_bias
        N_Hidden_Layers=hl_v
        lst_neorns=neo_v
        num_iters=epochs
        lr=LR
        parameters=model(x,y,N_Hidden_Layers,lst_neorns,num_iters,lr,activ,addbias)
        ###(x_train,y_train,Add_bias,LR,epochs,activ_func,hl_v,neo_v)
        
        print("Parameters :",parameters)

        ##train accuracy
        
        pred_train= predict_soft_max(parameters, x,activ,addbias)
       
        acc_tr=confustion_matrix_for_multi(y,pred_train)
        print(f"Training AVERAGE ACCURACY ={acc_tr} %")
        
        #test accuracy
        pred_test= predict_soft_max(parameters, xt,activ,addbias)
        
        acc_ts=confustion_matrix_for_multi(yt,pred_test)
        print(f"Testing AVERAGE ACCURACY ={acc_ts} %")
        
        
        Output.insert(tk.END,f" Confusion_acc_Testing_set = {acc_ts}%")
        print("Code is Running")
        

    B=tk.Button(top,text='RUN',height=3,width=8,command=callback_Run)
    B.place(x=200,y=350)

    Output = tk.Text(top, height = 3,
                  width = 40,
                  bg = "light cyan")
    Output.place(x=100,y=425)
    
    
    top.mainloop()


# In[42]:


def main():
    exe_GUI()

if __name__=='__main__':
    main()


# In[ ]:




