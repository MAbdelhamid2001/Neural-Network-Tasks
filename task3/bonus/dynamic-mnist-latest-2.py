#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder ,MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf


# In[3]:


data = pd.read_csv(r'F:\mohamed\4th_year_new\NN\labs\mnist_train.csv')
test = pd.read_csv(r'F:\mohamed\4th_year_new\NN\labs\mnist_test.csv')


# In[4]:


data


# In[5]:


def preprocessing_mnist(data,test):
    
    x=data.drop(['label'],axis=1)
    y=data['label']
    
    x=np.array(x)
    y=np.array(y)
    
    xt=test.drop(['label'],axis=1)
    yt=test['label']
    
    xt=np.array(xt)
    yt=np.array(yt)
    
    
    x,xt=x/255.0,xt/255.0

    #one hot Encoding for multi class
    y=pd.get_dummies(y)
    y=np.float32(y)

    yt=pd.get_dummies(yt)
    yt=np.float32(yt)

    print("train data")
    print("x",x.shape)
    print("y",y.shape)

    #test data
    print("test data")
    print("x",xt.shape)
    print("y",yt.shape)

    return x,xt,y,yt


# In[6]:


x,xt,y,yt=preprocessing_mnist(data,test)


# In[7]:


def initialize_params(i_l,N_Hidden_Layers,lst_neorns,o_l,Add_bias):
    
    
    NH_layers=N_Hidden_Layers
    
    lst_w_hid =[]
    lst_b_hid =[]
    if Add_bias==True:
    
        w_input=np.random.rand(i_l,lst_neorns[0])#0 ---> layer 1
        b_input=np.zeros((1,lst_neorns[0]))
        #########

        for i in range(0,NH_layers-1):
            curw=np.random.rand(lst_neorns[i],lst_neorns[i+1])
            curb=np.zeros((1,lst_neorns[i+1]))
            lst_w_hid.append(curw)
            lst_b_hid.append(curb)

        w_out=np.random.rand(lst_neorns[-1],o_l)
        b_out=np.zeros((1,o_l))


        par={"W1":w_input,
             "b1":b_input,
             "W_hid":lst_w_hid,
             "b_hid":lst_b_hid,
            "W3":w_out,
            "b3":b_out}
        return par
    
    else:#if select  no bias
        
        w_input=np.random.rand(i_l,lst_neorns[0])#0 ---> layer 1
        #########

        for i in range(0,NH_layers-1):
            curw=np.random.rand(lst_neorns[i],lst_neorns[i+1])
            lst_w_hid.append(curw)

        w_out=np.random.rand(lst_neorns[-1],o_l)

        par={"W1":w_input,
             "W_hid":lst_w_hid,
            "W3":w_out}
        return par


# In[8]:


def sigmoid(z):
    p=1/(1+np.exp(-z))
    return p


# In[9]:


def d_sigmoid(z):
    return (z*(1-z))


# In[10]:


def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))


# In[11]:


def d_tanh(z):
    return 1-z**2 


# In[12]:


def error(a3,y):#cost
    pred=a3
    m=y.shape[0]
#     epsilon = 1e-5 # error happend cause log doesnt accept 0 so i add 1e-5 to 0
#     cost = -np.sum(Y * np.log(A2 + epsilon) + (1 - Y) * np.log(1 - A2 + epsilon)) / m 
    cost=(1/(2*m))*np.sum(np.square(pred-y))
    return cost


# In[13]:


def forward_propagation(par,x,activ,Add_bias):

    if activ =='sigmoid':
        activation=sigmoid
        der_activ =d_sigmoid
    else:
        activation=tanh
        der_activ=d_tanh
    
    
    if Add_bias==True:
        
        w1=par["W1"]
        b1=par["b1"]
        ######
        w_hid_lst=par["W_hid"]##list
        b_hid_lst=par["b_hid"]##list
        #####
        w3=par["W3"]
        b3=par["b3"]

        z1=np.dot(x,w1)+b1
        a1=activation(z1)
        temp = a1

        z_hidd=[]
        a_hidd=[]

        for i in range(len(w_hid_lst)):
            z_hidd.append(np.dot(temp,w_hid_lst[i])+b_hid_lst[i])
            a_hidd.append(activation(z_hidd[-1]))
            temp = a_hidd[-1]


        z3=np.dot(temp,w3)+b3
        a3=tf.nn.softmax(z3)

        cache = {
             "a1": a1,
             "a_hid": a_hidd,
             "a3": a3
                }
        return a3,cache ,der_activ
    
    else:
        
        w1=par["W1"]
        ######
        w_hid_lst=par["W_hid"]##list
        #####
        w3=par["W3"]

        z1=np.dot(x,w1)
        a1=activation(z1)
        temp = a1

        z_hidd=[]
        a_hidd=[]

        for i in range(len(w_hid_lst)):
            z_hidd.append(np.dot(temp,w_hid_lst[i]))
            a_hidd.append(activation(z_hidd[-1]))
            temp = a_hidd[-1]


        z3=np.dot(temp,w3)
        a3=tf.nn.softmax(z3)

        cache = {
             "a1": a1,
             "a_hid": a_hidd,
             "a3": a3
                }
        return a3,cache ,der_activ


# In[14]:


def back_propagation(cache,x,y,par,der_activ):

    w1=par["W1"]
    w2=par["W_hid"]
    w3=par["W3"]

    A1 = cache['a1']#y1
    A2 = cache['a_hid']#y2
    A3 = cache['a3']#y3

    dA3=der_activ(A3) # derivative
    error3=dA3*(y-A3)
    ###
    error2 = []
    
    temp = error3
    en = 0
    idx = -1 
    A2 = A2[::-1]
    for i in A2:
        dA = der_activ(i)#derivative
        cur_w = -1
        if en == 0 : cur_w = w3.T
        else : 
            cur_w = w2[idx].T
            idx-=1
        error2.append(dA * np.dot(temp,cur_w))
        en = 1
        temp = error2[-1]

        
    error2 = error2[::-1]


    dA1=der_activ(A1)#derivative
    error1=dA1*np.dot(error2[0],w2[idx].T)

    ers = {"er1": error1,
             "er2": error2,
           "er3": error3,
          }
    return ers


# In[15]:


def weights_adaptation(par,er,LR,x,cache,Add_bias):

    if Add_bias ==True:

        W1 = par["W1"]
        b1 = par["b1"]

        W2 = par["W_hid"]#w of hidden
        b2 = par["b_hid"]#b of hidden

        W3 = par["W3"]
        b3 = par["b3"]

        er1 = er["er1"]
        er2 = er["er2"]#er of hidden
        er3 = er["er3"]

        A1 = cache['a1']#y1
        A2 = cache['a_hid']#y2 of hidden
        A3 = cache['a3']#y3

        W1 = W1 + LR*np.dot(x.T,er1)
        b1 = b1 + LR*np.sum(er1, axis=0)

        w_updated = []
        b_updated = []
        idx = 0
        en = 0
        for i in range(len(W2)):
            A = -1
            if en == 0 :
                A = A1.T
            else : 
                A = A2[idx].T
                idx += 1
            w_updated.append(W2[i] + LR * np.dot(A,er2[i]))
            b_updated.append(b2[i] + LR * np.sum(er2[i], axis=0))
            en = 1


        W3 = W3 + LR*np.dot(A2[-1].T,er3)
        b3 = b3 + LR* np.sum(er3, axis=0)

        WEIGHTS= {"W1": W1,
                      "b1": b1,
                      "W_hid": w_updated,
                      "b_hid": b_updated,
                      "W3": W3,
                      "b3": b3
                 }
        return WEIGHTS
    
    else:
        
        W1 = par["W1"]
        W2 = par["W_hid"]#w of hidden
        W3 = par["W3"]

        er1 = er["er1"]
        er2 = er["er2"]#er of hidden
        er3 = er["er3"]

        A1 = cache['a1']#y1
        A2 = cache['a_hid']#y2 of hidden
        A3 = cache['a3']#y3

        W1 = W1 + LR*np.dot(x.T,er1)

        w_updated = []
        idx = 0
        en = 0
        for i in range(len(W2)):
            A = -1
            if en == 0 :
                A = A1.T
            else : 
                A = A2[idx].T
                idx += 1
            w_updated.append(W2[i] + LR * np.dot(A,er2[i]))
            en = 1


        W3 = W3 + LR*np.dot(A2[-1].T,er3)

        WEIGHTS= {"W1": W1,
                      "W_hid": w_updated,
                      "W3": W3,
                 }
        return WEIGHTS


# In[17]:


def model(x,y,N_Hidden_Layers,lst_neorns,num_iters,lr,activ,Add_bias=1):
    
    inl=x.shape[1]
    outl=y.shape[1]
    par=initialize_params(inl,N_Hidden_Layers,lst_neorns,outl,Add_bias)
    costs=[]
    LR=lr
    

    for i in range(0, num_iters):
        a3,cache ,der_activ =forward_propagation(par,x,activ,Add_bias)

        cost=error(a3,y)

        
        ers=back_propagation(cache,x,y,par,der_activ) #DW derivatives
###
###
        par=weights_adaptation(par,ers,LR,x,cache,Add_bias)

        if i%100==0:
            costs.append(cost)
            print ("Cost after iteration %i: %f" %(i, cost))
    return par


# In[19]:


#parameters=model(5,2,[6,10],3,num_iters=10000,lr=0.001)
#model(784,10,10,10,num_iters=1000,lr=0.9) #3.24 acc
#(784,5,[5,6,5,6,5],10,num_iters=1000,lr=0.000005)#0.449
activ='tanh'
addbias=10
N_Hidden_Layers=10
lst_neorns=[10,10]
num_iters=500
lr=0.001
parameters=model(x,y,N_Hidden_Layers,lst_neorns,num_iters,lr,activ,addbias)


# In[87]:


parameters


# In[93]:


def predict_soft_max(par, x,activ,Add_bias):
    
    a3,cache ,der_activ = forward_propagation(par,x,activ,Add_bias)
    props_pred=a3.numpy()
    y_pred=np.zeros(a3.shape)
    for row in range(len(props_pred)):
        idx=np.argmax(props_pred[row])
        y_pred[row][idx]=1
    return  y_pred


# In[94]:


#confusion for multi class classification  
def confustion_matrix_for_multi(y,predictions):
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    confusion_matrix=confusion_matrix(y.argmax(axis=1), predictions.argmax(axis=1))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()
    plt.show()

    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Overall accuracy
    accuracy = (TP + TN) / (TP+ TN + FP + FN)
    print("confusion matrix is done : Accuracy per class ={}% , tp ={} , tn ={} ,fp ={} ,fn ={}".format(accuracy*100,TP,TN,FP,FN))
    print('*'*50)
    print(f"AVERAGE ACCURACY ={round(np.mean(accuracy)*100,3)} %")


# In[99]:


predictions = predict_soft_max(parameters, xt,activ,addbias)


# In[92]:


confustion_matrix_for_multi(y,predictions)

