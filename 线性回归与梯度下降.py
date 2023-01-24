#!/usr/bin/env python
# coding: utf-8

# In[255]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
# 糖尿病数据集
from sklearn.datasets import load_diabetes
#没使用
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()


# In[254]:


class LinearRegression:
    def __init__(self,data,labels,polynomial_degree=0,sinussoid_degree=0,normalize_data=True):
        self.data = data
        self.label = labels
        self.mean_data = data.mean()
        self.var_data = data.var()
        num_features = data.shape[1]
        self.theta = np.zeros((num_features,1))
        
    def train(self,alpha,num_interation=500):
        cost_history = self.gradient_descent(alpha,num_interation)
        return self.theta,cost_history
    def gradient_descent(self,alpha,num_interation):
        cost_histroy = []
        for _ in range(num_interation):
            self.gradient_step(alpha)
            cost_histroy.append(self.cost_function(self.data,self.label))
        return cost_histroy
        #该方法用来计算梯度下降的步骤，主要是theta的更新
        #alpha是学习率，也就是步长
    def gradient_step(self,alpha):
        #样本数量
        num_examples = self.data.shape[0]
        #预测值计算，调用同一个类下的方法
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        #误差值，也就是使用预测值减去真实值
        delta = prediction - self.label
        theta = self.theta
        #theta值的更新迭代
        self.theta = theta - alpha*(1/num_examples)*np.dot(delta.T,self.data).T#误差矩阵要进行转置
        #用来计算预测值的方法
    def hypothesis(data,theta):
        prediction = np.dot(data,theta)
        return prediction
    def cost_function(self,data,label):
        num = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - label
        cost = (1/2)*np.dot(delta.T,delta)
        return cost[0][0]
    def predict(self,data_process):
        predictions = LinearRegression.hypothesis(data_process,self.theta)


# In[158]:


diabetes = load_diabetes()
X = diabetes.data           # data
y = diabetes.target         # label
y=y.reshape(442,1)


# In[159]:


print(y.shape)


# In[160]:


example = LinearRegression(X,y)


# In[161]:


theta,cost_history = example.train(64,50000)


# In[162]:


theta


# In[163]:


cost_history[0]


# In[164]:


cost_history[-1]


# In[252]:


plot.plot(range(50000),cost_history)


# In[239]:


import random
gama = np.random.randn(1,100)
X_new_1 = np.arange(0,100).reshape(1,100).T
X_new_2 = np.arange(0,200,2).reshape(1,100).T
X_new = np.append(X_new_1,X_new_2,axis = 1)
Y_new = 5*X_new_1+10*X_new_2+10*gama.T


# In[240]:


plot.scatter(X_new_1,Y_new)


# In[247]:


example = LinearRegression(X_new,Y_new)
theta_new,cost_history_new = example.train(0.0000005,1000)


# In[248]:


print(theta_new)
print(cost_history_new[0])
print(cost_history_new[-1])


# In[253]:


plot.plot(range(1000),cost_history_new)


# In[ ]:




