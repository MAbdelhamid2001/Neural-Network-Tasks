#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tkinter as tk
from tkinter import messagebox,ttk
from Adaline_task import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
def exe_GUI():

    ###
#     data=pd.read_csv(r'F:\mohamed\4th year new\NN\labs\Lab3 (1)\penguins.csv')
#     data=preprocessing(data)

    ###
    top=tk.Tk()
    top.geometry('500x500')
    top.minsize(500, 500) 
    top.maxsize(500, 500)
    top.title('Perceptron Algorithm GUI')
    top['background']='#856ff8'

    ###activation
    act_l=tk.Label(top,text="Active_fun",width=8,height=2)
    act_l.place(x=90,y=0)

    act_var=tk.StringVar()
    act=ttk.Combobox(top,textvariable=act_var,width=20)
    act['values']=('Signnum','Linear')
    act.current(1)
    act.place(x=180,y=0)
    
    def retrieve_act():
        act_v=act.get()
        print(act_v)
        return(act_v)
    
    Button = tk.Button(top, text = "Submit", command = retrieve_act,width=8,height=1)
    Button.place(x=330,y=0)
    
    
    ###features
    ###f1
    f1_l=tk.Label(top,text="Feature 1",width=8,height=2)
    f1_l.place(x=90,y=50)

    f1_var=tk.StringVar()
    f1=ttk.Combobox(top,textvariable=f1_var,width=20)
    f1['values']=('bill_length_mm','bill_depth_mm','flipper_length_mm','gender','body_mass_g')
    f1.current(0)
    f1.place(x=180,y=50)
    
    def retrieve_f1():
        f1_v=f1.get()
        print(f1_v)
        return(f1_v)
    
    Button = tk.Button(top, text = "Submit", command = retrieve_f1,width=8,height=1)
    Button.place(x=330,y=50)

    ###f2
    f2_l=tk.Label(top,text="Feature 2",width=8,height=2)
    f2_l.place(x=90,y=100)


    f2_var=tk.StringVar()
    f2=ttk.Combobox(top,width=20,textvariable=f2_var)
    f2['values']=('bill_length_mm','bill_depth_mm','flipper_length_mm','gender','body_mass_g')
    f2.current(1)
    f2.place(x=180,y=100)

    def retrieve_f2():
        f2_v=f2.get()
        print(f2_v)
        return f2_v
    Button = tk.Button(top, text = "Submit", command = retrieve_f2,width=8,height=1)
    Button.place(x=330,y=100)

    ##classes

    ###c1
    c1_l=tk.Label(top,text="Class 1",width=8,height=2)
    c1_l.place(x=90,y=150)


    c1_var=tk.StringVar()
    c1=ttk.Combobox(top,width=20,textvariable=c1_var)
    c1['values']=('Adelie', 'Chinstrap', 'Gentoo')
    c1.current(2)
    c1.place(x=180,y=150)
    ('Adelie', 'Chinstrap', 'Gentoo')
    def retrieve_c1():
        c1_v=c1_var.get()
        print(c1_v)
        return c1_v
    
    Button = tk.Button(top, text = "Submit", command = retrieve_c1,width=8,height=1)
    Button.place(x=330,y=150)

    ###c2
    c2_l=tk.Label(top,text="Class 2",width=8,height=2)
    c2_l.place(x=90,y=200)

    c2_var=tk.StringVar()
    c2=ttk.Combobox(top,width=20,textvariable=c2_var)
    c2['values']=('Adelie', 'Chinstrap', 'Gentoo')
    c2.current(1)
    c2.place(x=180,y=200)

    def retrieve_c2():
        c2_v=c2_var.get()
        print(c2_v)
        return c2_v
    
    Button = tk.Button(top, text = "Submit", command = retrieve_c2,width=8,height=1)
    Button.place(x=330,y=200)


    # #entr Learning Rate
    lr=tk.Label(top,text="LearnRate",width=8,height=2)
    lr.place(x=90,y=250)
    lr_entry=tk.Entry(top,width=20)
    lr_entry.place(x=180,y=250)
    #lr_entry.focus_set()
    lr_entry.insert(0,0.5)
    def callback_lr():
        lr_v=lr_entry.get()
        print(lr_v)
        return lr_v
    
    B=tk.Button(top,text='insert',command=callback_lr,width=8,height=1)
    B.place(x=330,y=250)


    ##enter epochs
    ep=tk.Label(top,text="Epochs",width=8,height=2)
    ep.place(x=90,y=300)
    ep_entry=tk.Entry(top,width=20)
    ep_entry.place(x=180,y=300)
    ep_entry.insert(0,20)

    #ep_entry.focus_set()
    def callback_ep():
        ep_v=ep_entry.get()
        print(ep_v)
        return ep_v

    B=tk.Button(top,text='insert',command=callback_ep,width=8,height=1)
    B.place(x=330,y=300)


    ####
    ##enter thresh
    th=tk.Label(top,text="Thresh",width=8,height=2)
    th.place(x=90,y=350)
    th_entry=tk.Entry(top,width=20)
    th_entry.place(x=180,y=350)
    th_entry.insert(0,0.2)

    #ep_entry.focus_set()
    def callback_thresh():
        th_v=th_entry.get()
        print(th_v)
        return th_v

    B=tk.Button(top,text='insert',command=callback_thresh,width=8,height=1)
    B.place(x=330,y=350)
    ###3#
        


    ##Addbias
    bias_l=tk.Label(top,text="Bias",width=8,height=2)
    bias_l.place(x=90,y=400)

    def add_bias():
        b_v=check_bias.get()
        print(f'bias state{b_v}')
        return b_v
    
    check_bias=tk.IntVar()
    c1=tk.Checkbutton(top,variable=check_bias,onvalue=1,offvalue=0,command=add_bias)
    c1.place(x=180,y=400)
    check_bias.get()

    ##Run
    def callback_Run():
        msg=messagebox.showinfo('welcome','Code is Running')
        print("Code is Running")

        Output.delete('1.0', tk.END)
        f1_v= retrieve_f1()
        f2_v= retrieve_f2()
        c1_v=retrieve_c1()
        c2_V=retrieve_c2()
        LR=float(callback_lr())
        epochs=int(callback_ep())
        Add_bias=add_bias()
        activ_func=retrieve_act()

        thres=float(callback_thresh())
        if activ_func =='signnum':
            activ_func=signnum
        else:
            activ_func=linear##to be linear
        print(f1_v,f2_v,c1_v,c2_V,epochs,LR,Add_bias)
        #thresh=0.5#####
        data=pd.read_csv(r'F:\mohamed\4th_year_new\NN\labs\Lab3 (1)\penguins.csv')
        data=preprocessing(data)
        
        #draw combinations of features
        
#         plt.figure(1)
#         print(sns.pairplot(data,hue='species'))
#         plt.show()
        
        x_train,x_test,y_train,y_test=get_selected_data(f1_v,f2_v,c1_v,c2_V,data=data)
        
        x_train=normalize(x_train)
        x_test=normalize(x_test)
        
        w_ ,b_=perceptron_algorithm(x_train,y_train,Add_bias,LR,epochs,activ_func,thresh=thres)
        print("weights : w={} ,b={}".format(w_,b_))

        ##train accuracy
        pred_train=testing(x_train,w_,b_,activ_func)
        print(f'Final training accuracy score {accuracy_score(y_train,pred_train)*100}%')
        
        #test accuracy
        pred_test=testing(x_test,w_,b_,activ_func)
        print(f'Final testing accuracy score {accuracy_score(y_test,pred_test)*100}%')
        
        
        accuracy=confustion_matrix(y_test, pred_test)
        
        
        Output.insert(tk.END,f" Confusion_acc_Testing_set = {round(accuracy,2)*100}%")
        print("Code is Running")


        print(draw_decision_boundary_3(x_test,y_test,w_,b_,"Testing Data"))
        plt.show()
        ###########
        

    B=tk.Button(top,text='RUN',height=2,width=7,command=callback_Run)
    B.place(x=250,y=400)

    Output = tk.Text(top, height = 2,
                  width = 30,
                  bg = "light cyan")
    Output.place(x=100,y=450)
    
    
    top.mainloop()


# In[42]:


def main():
    exe_GUI()

if __name__=='__main__':
    main()

