# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:33:47 2021
At University of Toronto
@author: Sheng Shi
"""

import torch
import torch.nn as nn


# Define NN models ------------------------------------------------------------
class MyNet_MLP1(nn.Module): # Single hidden layer
    def __init__(self,n_input,n_hidden1,n_output):
        super().__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden1)
        self.predict = nn.Linear(n_hidden1,n_output)
    def forward(self,input):
        out = input.view(-1,input.size(1)*input.size(2)*input.size(3))
        out = self.hidden1(out)
        out = nn.LeakyReLU(negative_slope=0.05)(out)
        out =self.predict(out)
        return out


# Test models -----------------------------------------------------------------
def Test_MyNet(net,test_x,test_y):
    net.eval()
    with torch.no_grad():
        test_output = net(test_x)
        test_err = (test_y-test_output)
    return test_err