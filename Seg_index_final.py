# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:41:56 2017

@author: KUMARPX21
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:19:26 2017

@author: KUMARPX21
"""

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import stats
from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
   
import math
from pylab import *

plt.style.use('seaborn-white')

#Loading Data
mat_contents=sio.loadmat('input_mar_18.mat')
#C=mat_contents['Address']
Input=mat_contents['Data']
Input[:,10]=0
#Data_fields=mat_contents['Data_fields']
Input=np.matrix(Input)

ct=-1
Data=np.zeros((len(Input),9))
Response=np.zeros((len(Input),1))
R1=mat_contents['Features']
R2=mat_contents['Response']
complete_sim=R1[:,1] # Complete/Incomplete simulations

(m,n)=R1.shape
for i in range(0,len(complete_sim)):
    if(complete_sim[i]==1 and R1[i,2]==0):
        ct=ct+1;
        Data[ct,:]=R1[i,3:12] # Features
        Response[ct]=R2[i,0] # Mixing Time
Data=Data[0:ct+1,:]
Response=Response[0:ct+1]
    
Data=np.asmatrix(Data)

raw_data = Data;
#Data=stats.zscore(Data,axis=0,ddof=1)
raw_response = Response

#################################################3
(p,q)=Data.shape
MSE_svr = MSE_pcr = MSE_plsr = MSE_lr = 0
nlambda = 100
MSE_en = np.zeros((nlambda,1))
coeff_en = np.zeros((nlambda,q))
Pred_en = np.zeros((nlambda,p))
Pred_test = np.zeros((200,p))
training = np.zeros((p-1,1))
Y_train=np.zeros((p-1,1))
X_train=np.zeros((p-1,q))
X_test=np.zeros((1,q))
Coeff_plsr=np.zeros((len(Data[0]),1))
XX=1

if(XX==1):
    for i in range(0,len(Data)):
        testing=i;
    
        ct = -1
        for k in range(0,len(Data)-1):
            if(k!=i):
                ct=ct+1
                training[ct,0]=int(k)
                Y_train[ct,0]=Response[int(k)]
                X_train[ct,:]=Data[int(k),:]
    
        mean_data = np.mean(X_train,axis=0)
        std_data = np.std(X_train,axis=0)     
        #Data=stats.zscore(Data,axis=0,ddof=1)
        X_train=(X_train-mean_data)/std_data
        X_train=np.nan_to_num(X_train)
        
        mean_resp = np.mean(Y_train,axis=0)
        std_resp = np.std(Y_train,axis=0)
        Y_train=(Y_train-mean_resp)/std_resp
        Y_train=np.nan_to_num(Y_train)
        
        Y_test=Response[int(i)]
        X_test=(Data[int(i),:]-mean_data)/std_data
    
        Pred_test[0,i]=Y_test
                 
        # Linear Regression
        lr = linear_model.LinearRegression()
        mdl_lr = lr.fit(X_train,Y_train)
        Coeff_LR = mdl_lr.coef_
        Pred_test[1,i] = mdl_lr.predict(X_test)*std_resp+mean_resp
        MSE_lr = MSE_lr + abs(Y_test-Pred_test[1,i])
        
        # Support Vector Machine Regression
        parameters1={'C':[1,1]}
        svr_lin = SVR(kernel='linear')
        svr_lin = GridSearchCV(svr_lin,parameters1)
        svr_rbf = SVR(kernel='rbf',C=1,gamma=1)
        mdl_svr = svr_lin.fit(X_train,Y_train)
        mdl_nsvr = svr_rbf.fit(X_train,Y_train)
        Pred_test[2,i]=mdl_svr.predict(X_test)*std_resp + mean_resp          
        Pred_test[7,i]=mdl_nsvr.predict(X_test)*std_resp + mean_resp          
        MSE_svr = MSE_svr + abs(Pred_test[1,i] - Y_test)
        
        # Regression Tree with Random Forest
        max_depth = 7
        parameters2 = {'n_estimators':[5,10,15,25], 'max_depth':[5,7,10]}
        mdl_regrf = RandomForestRegressor(random_state=0,max_features='auto',warm_start='true',bootstrap='true')
        mdl_regrf = GridSearchCV(mdl_regrf,parameters2)
        mdl_regrf.fit(X_train, Y_train)
        # Predict on test data
        Pred_test[3,i] = mdl_regrf.predict(X_test)*std_resp + mean_resp
        
        # Decision Tree Regressor
        parameters3={'max_depth':[5,7]}
        mdl_dt = DecisionTreeRegressor(random_state=0)
        mdl_dt = GridSearchCV(mdl_dt,parameters3)
        
        parameters4={'n_estimators':[10,25]}
        mdl_adadt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),random_state=0)
        mdl_adadt = GridSearchCV(mdl_adadt,parameters4)
        
        y_dt=mdl_dt.fit(X_train, Y_train)
        y_adadt=mdl_adadt.fit(X_train, Y_train)
        # Predict on test data
        Pred_test[4,i] = mdl_dt.predict(X_test)*std_resp + mean_resp
        Pred_test[5,i] = mdl_adadt.predict(X_test)*std_resp + mean_resp
        
        # Elastic Net Regression
        Penalty = range(0,nlambda,1);
        for j in range(0,len(Penalty)):
            enet = ElasticNet(alpha=0.01,l1_ratio=Penalty[j]/nlambda)
            mdl_en = enet.fit(X_train,Y_train)
            Pred_en[j,i] = mdl_en.predict(X_test)*std_resp+mean_resp
            coeff_en[j,:]=coeff_en[j,:] + enet.coef_
            MSE_en[j,0] = MSE_en[j,0] + abs(mdl_en.predict(X_test)*std_resp+mean_resp - Y_test)
        
        # Partial Least Squares Regression
        mdl_plsr=PLSRegression(n_components=5)
        mdl_plsr.fit(X_train, Y_train)
        Pred_test[8,i] = mdl_plsr.predict(X_test)*std_resp + mean_resp
        Coeff_plsr=Coeff_plsr+mdl_plsr.coef_
        
    MSE_en = MSE_en/p
    MSE_svr = MSE_svr/p
    MSE_lr = MSE_lr/p
    coeff_en = coeff_en/(p)
    Coeff_plsr = Coeff_plsr/len(Data)
    
    min_en = np.argmin(np.min(MSE_en,axis=1))
    Pred_test[6,:]=Pred_en[min_en,:]
    Coeff_ENet = coeff_en[min_en,:]
    
    r2_score_lr = r2_score(Pred_test[0,:], Pred_test[1,:])
    r2_score_svr = r2_score(Pred_test[0,:], Pred_test[2,:])
    r2_score_svr = r2_score(Pred_test[0,:], Pred_test[3,:])
    r2_score_dt = r2_score(Pred_test[0,:], Pred_test[4,:])
    r2_score_adadt = r2_score(Pred_test[0,:], Pred_test[5,:])
    r2_score_svr = r2_score(Pred_test[0,:], Pred_test[6,:])
    r2_score_nsvr = r2_score(Pred_test[0,:], Pred_test[7,:])
    r2_score_plsr = r2_score(Pred_test[0,:], Pred_test[8,:])
    
    params = {
       'axes.labelsize': 22,
       'text.fontsize': 14,
       'legend.fontsize': 16,
       'xtick.labelsize': 16,
       'ytick.labelsize': 16,
       'text.usetex': False,
       'figure.figsize': [10, 10]
       }
    rcParams.update(params)
    fig1=plt.figure(dpi=150, facecolor='w', edgecolor='w')
    
    #plt.plot(Pred_test[0,:],Pred_test[1,:],'.',c='b',label='LR: $R^2$ = %.2f' %r2_score_lr)
    plt.subplot(221)
    plt.plot(Pred_test[0,:],Pred_test[2,:],'.',ms=8,c='b',label='SVR: $R^2$ = %.2f' %r2_score_svr)
    plt.plot(Pred_test[0,:],Pred_test[0,:],c='k',label='Y=X',linewidth=1)
    plt.legend(loc='best',fontsize=10)
    plt.ylabel('Machine Learning Predicted (sec)',fontweight='bold',fontsize=14)
    plt.xlabel('DEM Calculated (sec)',fontweight='bold',fontsize=14)
    plt.axis([0,200,0,200])
    plt.tick_params(labelsize=10)
    
    plt.subplot(222)
    plt.plot(Pred_test[0,:],Pred_test[3,:],'.',ms=8,c='b',label='RT-RF: $R^2$ = %.2f' %r2_score_regrf)
    plt.plot(Pred_test[0,:],Pred_test[0,:],c='k',label='Y=X',linewidth=1)
    plt.legend(loc='best',fontsize=10)
    plt.ylabel('Machine Learning Predicted (sec)',fontweight='bold',fontsize=14)
    plt.xlabel('DEM Calculated (sec)',fontweight='bold',fontsize=14)
    plt.axis([0,200,0,200])
    plt.tick_params(labelsize=10)
    
    plt.subplot(223)
    plt.plot(Pred_test[0,:],Pred_test[6,:],'.',ms=8,c='b',label='EN: $R^2$ = %.2f' %r2_score_en)
    plt.plot(Pred_test[0,:],Pred_test[0,:],c='k',label='Y=X',linewidth=1)
    plt.legend(loc='best',fontsize=10)
    plt.ylabel('Machine Learning Predicted (sec)',fontweight='bold',fontsize=14)
    plt.xlabel('DEM Calculated (sec)',fontweight='bold',fontsize=14)
    plt.axis([0,200,0,200])
    plt.tick_params(labelsize=10)
    
    plt.subplot(224)
    plt.plot(Pred_test[0,:],Pred_test[8,:],'.',ms=8,c='b',label='PLSR: $R^2$ = %.2f' %r2_score_plsr)
    #plt.plot(Pred_test[0,:],Pred_test[4,:],'P',ms=10,c='m',label='DT: $R^2$ = %.2f' %r2_score_dt)
    plt.plot(Pred_test[0,:],Pred_test[0,:],c='k',label='Y=X',linewidth=1)
    plt.legend(loc='best',fontsize=10)
    plt.ylabel('Machine Learning Predicted (sec)',fontweight='bold',fontsize=14)
    plt.xlabel('DEM Calculated (sec)',fontweight='bold',fontsize=14)
    plt.axis([0,200,0,200])
    plt.tick_params(labelsize=10)
    
    plt.grid(False)
    plt.show()
    fig1.savefig('tmix_ML.tif',facecolor='white', edgecolor='white')

#Pred_test=np.matrix.transpose(Pred_test)

# Running the best model on the entire data set==> RT-RF
mean_data1 = np.mean(Data,axis=0)
std_data1 = np.std(Data,axis=0) 
Data=(Data-mean_data1)/std_data1
Data=np.nan_to_num(Data)
mean_resp1 = np.mean(Response,axis=0)
std_resp1 = np.std(Response,axis=0)
Response=(Response-mean_resp1)/std_resp1

# Finding the best parameters
max_depth = 5
parameters2 = {'n_estimators':[10,15,20,25], 'max_depth':[3,4,5]}
mdl_regrf = RandomForestRegressor(random_state=0,max_features='auto',warm_start='true',bootstrap='true')
mdl_regrf = GridSearchCV(mdl_regrf,parameters2)
mdl_regrf.fit(Data,Response)
print(mdl_regrf.best_params_)

# Fitting with the best parameters
mdl_regrf = RandomForestRegressor(random_state=0,max_features='auto',warm_start='true',bootstrap='true',max_depth=3,n_estimators=25)
mdl_regrf.fit(Data,Response)
Feature_Selection=mdl_regrf.feature_importances_

