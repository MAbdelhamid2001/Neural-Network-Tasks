#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# In[57]:


data=pd.read_csv(r'F:\mohamed\4th_year_new\NN\labs\Lab3 (1)\penguins.csv')


# In[58]:


def preprocessing(data):
    
    data.isnull().sum()

    mis=(data['gender'].isnull().sum()/len(data))*100
    print(f'Percentage of missing values at gender column = {mis} %')

## so we will keep the column and fill the missing data with mode of the data

    #data['gender'].mode()

    data['gender'].fillna(data['gender'].mode()[0],inplace=True)

    #print(sns.pairplot(data,hue='species'))

# Features `bill_depth_mm` and `body_mass_g` between (Gentoo,Adelie) or (Gento,chainstrap) Classes
# Features `bill_depth_mm` and `flipper_length_mm` between (Gentoo,Adelie) or (Gento,chainstrap) Classes
# These are best features to use ,we may get some small errors with other combinations of features .

## encoding gender and species columns

    data['gender']=pd.get_dummies(data['gender'],drop_first=True)

    
# use 30 for train and 20 for test for  each class
# so 60 randomly selected samples in train and 40 in test for all input data for 2 classes
    return data


# In[59]:
########



#########
def get_selected_data(f1,f2,c1,c2,data=data):
    
    df=data[['species',f1,f2]]
    d=df['species'].isin([c1,c2])
    dd=df.loc[d]
    
    le=LabelEncoder()
    dd['species']=le.fit_transform(dd['species'])
    
    x=dd.drop(['species'],axis=1)
    y=dd['species']
    y[y==0]=-1
    x=np.array(x)
    y=np.array(y)
    print(y)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,shuffle=True,stratify=y)
    
    return x_train,x_test,y_train,y_test


# In[60]:


# def step(z):
#     if z>=0:
#         return 1
#     else:
#         return 0


# In[61]:



# In[62]:

    
def signnum(z):
    #print("it works")
    if z>=0:
        return 1
    elif z<0:
        return -1


# In[63]:


def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
#
# def normalize(x):
#     return (x-np.mean(x))/np.std(x)

# ## Final perceptron algorithm

# In[64]:
def predict(x,w,b,activation):
    pred=np.dot(w,x)+b
    final_pred=activation(pred)
    return final_pred


def perceptron_algorithm(x_train,y_train,Add_bias,LR,epochs,activ_func):
    activation=activ_func
    if Add_bias==True:

        b=np.random.randn(1)
        w=np.random.randn(1,2)

        alpha=LR
        epochs=epochs

        for i in range(epochs):

            for x,y in zip(x_train,y_train):

                final_pred=predict(x,w,b,activation)
                error= y - final_pred
                if error !=0:
                    w=w+alpha*error*x.T
                    b=b+alpha*error
                    print("in add bias")
        return w ,b

    else:#if select  no bias

        b=0
        w=np.random.randn(1,2)
        alpha=LR
        epochs=epochs

        for i in range( epochs):

            for x,y in zip(x_train,y_train):
                y_pred=w.dot(x)

                final_pred=activation(y_pred)
                error= y - final_pred
                if error != 0:
                    w=w+alpha*error*x.T
                #b=b+alpha*error
        print("in add no bias")
        return w ,b

    
    
    
######################    

#
# def perceptron_algorithm(x_train,y_train,Add_bias,LR,epochs,activ_func):
#     activation=activ_func
#     if Add_bias==True:
#
#         b=np.random.randn(1)
#         w=np.random.randn(1,2)
#
#         alpha=LR
#         epochs=epochs
#
#         n_miss=1
#         iters = 0
#
#         h=0
#         m=0
#         w_h=0
#         b_h=0
#
#         while n_miss > 0 and iters < epochs:
#             iters +=1
#             n_miss =0
#             n_hit =0
#             for x,y in zip(x_train,y_train):
#                 y_pred=w.dot(x)+b
#                 final_pred=activation(y_pred)
#                 error= y - final_pred
#
#                 w=w+alpha*error*x.T
#                 b=b+alpha*error
#                 if y != final_pred:
#                     n_miss +=1
#                 else:
#                     n_hit +=1
#             h=n_hit
#             m=n_miss
#
#             if h>=m:
#                 w_h=w
#                 b_h=b
#
#
#
#             print(f'---------- Iteration No: {iters} ----------' )
#             print(f'Missed: {n_miss}')
#             print(f'successed: {n_hit}')
#             print(f'Training Accuracy: {((x_train.shape[0]-n_miss)/x_train.shape[0]) * 100}%')
#
#         #return w_ ,b_
#         if h>m:
#             w=w_h
#             b=b_h
#
#         return w ,b
#
#     else:
#
#         b=0
#         w=np.random.randn(1,2)
#
#         alpha=LR
#         epochs=epochs
#
#         n_miss=1
#         iters = 0
#         w_=0
#         b_=0
#
#         h=0
#         m=0
#         w_h=0
#         b_h=0
#         while n_miss > 0 and iters < epochs:
#             iters +=1
#             n_miss =0
#             n_hit =0
#             for x,y in zip(x_train,y_train):
#                 y_pred=w.dot(x)+b
#
#                 final_pred=activation(y_pred)
#                 error= y - final_pred
#
#                 w=w+alpha*error*x.T
#                 #b=b+alpha*error
#                 if y != final_pred:
#                     n_miss +=1
#                 else:
#                     n_hit +=1
#
#             h=n_hit
#             m=n_miss
#
#             if h>=m:
#                 w_h=w
#                 b_h=b
#
#             print(f'---------- Iteration No: {iters} ----------' )
#             print(f'Missed: {n_miss}')
#             print(f'successed: {n_hit}')
#             print(f'Training Accuracy: {((x_train.shape[0]-n_miss)/x_train.shape[0]) * 100}%')
#
#         #return w_ ,b_
#         if h>m:
#             w=w_h
#             b=b_h
#
#         return w ,b
#
#

#######################
# # testing

# In[65]:


#testing data
def testing(x_t,w_,b_,activ_func):
    pred_test=[]
    for x in x_t:
        final_pred=predict(x,w_,b_,activ_func)
        pred_test.append(final_pred)
    return pred_test

# - drwa line
# - confusuion matrix for testing and accuracy 
# - report with combination
# - ui separated from the code

# In[66]:


#new
def draw_decision_boundary_1(x_feat,y,w_,b_):
    x_=x_feat
    y_=y
       
    plt.scatter(x_[:,0],x_[:,1],c=y_)
    axes = plt.gca()
    
    ############
    w1=w_[0][0]
    w2=w_[0][1]
    b=b_
    c = -b/w2
    m = -w1/w2
    x_vals = np.array(axes.get_xlim())
    y_vals = c + m * x_vals
    plt.plot(x_vals, y_vals, '--')
    ##########
    plt.show()
# In[67]:

# #islam
def draw_decision_boundary_2(x_feat,y,w_,b_):
    pre=[]
    x_=x_feat
    y_=y    
    ############
    w1=w_[0][0]
    w2=w_[0][1]
    b=b_
    c = -b/w2
    m = -w1/w2
    x_vals = np.linspace( 0,0.1)
    y_vals = c + m * x_vals
    plt.plot(x_vals, y_vals, '--',color= 'red')
    ##########
    plt.scatter(x_[:,0],x_[:,1],c=y_)
    plt.show()

#new
def draw_decision_boundary_3(x_feat,y,w_,b_):
    x_=x_feat
    y_=y    
    ############
    w1=w_[0][0]
    w2=w_[0][1]
    b=b_
    c = -b/w2
    m = -w1/w2
    x_vals=np.linspace(np.amin(x_[:,0]),np.amax(x_[:,0]))
   # print(x_vals)
    y_vals = c + m * x_vals
    plt.plot(x_vals, y_vals, '--',color= 'red')
    ##########
    plt.scatter(x_[:,0],x_[:,1],c=y_)
    plt.show()

#es
def confustion_matrix(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
            FP += 1
        if y_actual[i]==y_pred[i]==-1:
            TN += 1
        if y_pred[i]==-1 and y_actual[i]!=y_pred[i]:
            FN += 1
    accuracy = (TP + TN) / (TP+ TN + FP + FN)
    print("confusion matrix is done : acurracy={} , tp ={} , tn ={} ,fp ={} ,fn ={}".format(accuracy,TP,TN,FP,FN))
    return accuracy