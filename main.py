# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:33:05 2021

@author: Sheng Shi
"""

import torch
import numpy as np
from utils_MyNet_train.DeepSVDD import DeepSVDD
from scipy import signal
from utils_MyNet_train.MyNet_train2 import Test_MyNet
from matplotlib import pyplot as plt
import pickle

# Hyper Parameters
window_size = 41
dt          = 4.16666667E-3
channel2x = (25,30,33,36,43,44); channel2y = (28,);   channel2z = (26,29,34,40,41)
channel3x = (17,19,39,46);       channel3y = (20,21); channel3z = (16,23,38)
channel = channel2x + channel2y + channel2z + channel3x + channel3y + channel3z
outputdim   = 4
kmlp        = window_size-1

# Transform data format
def ToNN(data,window_size): 
    prednum = [12,13,14,15]
    train = data
    data = np.zeros([train.shape[0]-window_size,window_size,train.shape[1],1])
    for i in range(len(train) - window_size):
        data[i,:,:,0] = train[(i+1): (i+window_size+1),:]
        data[i,:,prednum,0] = train[i: (i + window_size),prednum].T
    data = np.array(data).astype('float64')
    train_x = torch.from_numpy(data).permute(0,3,1,2)
    train_y = torch.from_numpy(train[window_size:,prednum])
    return train_x,train_y


# Load Predictor
print('Loading trained predictor ...')
predictor = torch.load('TrainedNeuralNetworks/predictor%d.pth' %(window_size))
# Generate innovations
print('Generating innovations ...')
## Undamaged feature for training (Trial 5)
data0 = np.loadtxt('TestingData/Trial5-Undamaged.csv', dtype="float", delimiter=',')[4000:70000,channel]
data0=signal.detrend(data0,axis=0)*4/3; data0=data0[:56000,:]
data1_1 = (data0[:28000,:])
data1_2 = (data0[40000:53000,:])
Train_x,Train_y = ToNN(data1_1,kmlp)
feature1_1 = Test_MyNet(predictor,Train_x,Train_y); feature1_1 = feature1_1.numpy()
## Undamaged feature for validation (Trial 5)
Valid_x,Valid_y = ToNN(data1_2,kmlp)
feature1_2 = Test_MyNet(predictor,Valid_x,Valid_y); feature1_2 = feature1_2.numpy()
## Undamaged feature for testing (Trial 5)
data1_3 = np.loadtxt('TestingData/Trial5-Undamaged.csv', dtype="float", delimiter=',')[4000:70000,channel]
data1_3 = signal.detrend(data1_3,axis=0)*4/3; data1_3=data1_3[63000:,:]
Test1_x,Test1_y = ToNN(data1_3,kmlp)
feature1_3 = Test_MyNet(predictor,Test1_x,Test1_y)
feature1_3 = feature1_3.numpy() 
## Damaged feature for testing (Trial 13)
data13 = np.loadtxt('TestingData/Trial13-AfterGilroy40%.csv', dtype="float", delimiter=',')[4000:70000,channel]
data13 = signal.detrend(data13,axis=0)*4/3
Test13_x,Test13_y = ToNN(data13,kmlp)
feature13 = Test_MyNet(predictor,Test13_x,Test13_y)
feature13 = feature13.numpy() 
## Damaged feature for testing (Trial 22)
data22 = np.loadtxt('TestingData/Trial22-AfterGilroy67%.csv', dtype="float", delimiter=',')[4000:70000,channel]
data22 = signal.detrend(data22,axis=0)
Test22_x,Test22_y = ToNN(data22,kmlp)
feature22 = Test_MyNet(predictor,Test22_x,Test22_y)
feature22 = feature22.numpy() 
## Damaged feature for testing (Trial 39)
data39 = np.loadtxt('TestingData/Trial39-AfterGilroy100%.csv', dtype="float", delimiter=',')[4000:70000,channel]
data39 = signal.detrend(data39,axis=0)
Test39_x,Test39_y = ToNN(data39,kmlp)
feature39 = Test_MyNet(predictor,Test39_x,Test39_y)
feature39 = feature39.numpy() 
## Damaged feature for testing (Trial 41)
data41 = np.loadtxt('TestingData/Trial41-AfterGilroy120%.csv', dtype="float", delimiter=',')[4000:70000,channel]
data41 = signal.detrend(data41,axis=0)
Test41_x,Test41_y = ToNN(data41,kmlp)
feature41 = Test_MyNet(predictor,Test41_x,Test41_y)
feature41 = feature41.numpy() 
    
# Load Decision-maker
print('Loading decisoin-maker ...')
DecisionMaker = DeepSVDD()
DecisionMaker.build_network(outputdim,2+outputdim,4+outputdim)
DecisionMaker.load_model('TrainedNeuralNetworks/DecisionMaker%d.pth' %(window_size))
R = DecisionMaker.R

# Calculate mfad
print('Start testing ...')
## Training mfad
threshold = np.array(0.0)
DecisionMaker.test((feature1_1))
score1_1 = DecisionMaker.test_scores
mfad1_1 = np.mean(score1_1>threshold)/dt
## Validation mfad
DecisionMaker.test((feature1_2))
score1_2 = DecisionMaker.test_scores
mfad1_2 = np.mean(score1_2>threshold)/dt
## Testing mfad
DecisionMaker.test((feature1_3))
score1_3 = DecisionMaker.test_scores
mfad1_3 = np.mean(score1_3>threshold)/dt
# Calculate mad
## Trial 13
DecisionMaker.test((feature13))
score13 = DecisionMaker.test_scores
mad13 = np.mean(score13>threshold)/dt
## Trial 22
DecisionMaker.test((feature22))
score22 = DecisionMaker.test_scores
mad22 = np.mean(score22>threshold)/dt
## Trial 39
DecisionMaker.test((feature39))
score39 = DecisionMaker.test_scores
mad39 = np.mean(score39>threshold)/dt
## Trial 41
DecisionMaker.test((feature41))
score41 = DecisionMaker.test_scores
mad41 = np.mean(score41>threshold)/dt

# Figures
print('Plotting figures ...')
## MAD
plt.figure(1)
plt.bar(['AfterGilroy40%','AfterGilroy67%','AfterGilroy100%','AfterGilroy120%'],[mad13,mad22,mad39,mad41])
plt.title('Results for MAD')
## MFAD
plt.figure(2)
plt.bar(['AfterGilroy40%','AfterGilroy67%','AfterGilroy100%','AfterGilroy120%'],[mfad1_3/mad13,mfad1_3/mad22,mfad1_3/mad39,mfad1_3/mad41])
plt.axhline(y=1, c="r", ls="--", lw=2)
plt.title('Results for MFAD/MAD')
## Validation loss
valid_loss_history = pickle.load(open('TrainedNeuralNetworks/valid_loss_history.pth','rb'))
nepochs = len(valid_loss_history)
valid_loss_all = []
for i in range(nepochs):
    valid_loss_temp = np.array(valid_loss_history[i])
    valid_loss_all.extend(valid_loss_temp)
plt.figure(3)
plt.plot(range(len(valid_loss_all)),valid_loss_all)
plt.axvline(x=86, c="r", ls="--", lw=2)
plt.axvline(x=157, c="r", ls="--", lw=2)
plt.axvline(x=242, c="r", ls="--", lw=2)
plt.axvline(x=301, c="r", ls="--", lw=2)
plt.axvline(x=354, c="r", ls="--", lw=2)
plt.title('Validation loss for iterative pruning')