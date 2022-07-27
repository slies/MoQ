#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
This code is the official Pytorch implementation of LSTNet (https://arxiv.org/pdf/1703.07015.pdf), please check the 
official documents for more detials.

https://github.com/fbadine/LSTNet           Tensorflow / Keras implementation

https://github.com/laiguokun/LSTNet         Pytorch implementation

"""


class LSTNet(nn.Module):
    """
    LSTNet is used as the expert of MoQ.
    
    (Section: Experts with Various Forecasting Styles)
    
    """
    
    def __init__(self, args):
        super(LSTNet, self).__init__()
        self.P = args.window                    # the length of input seq
        self.num_feat = args.num_feat           # the number of features the input seq
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip                   # The length of period to skip
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = args.highway_window           # the length of the window of highway network
        
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.num_feat), padding=self.Ck-1)
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = args.dropout)
        
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.num_feat)
        else:
            self.linear1 = nn.Linear(self.hidR, self.num_feat)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh
 
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, 1, self.P, self.num_feat)
        c = F.relu(self.conv1(c))
        c = c[:, :, :-(self.Ck-1), (self.Ck-1):-(self.Ck-1)]   # only keep the left padding, discard the right paddding.
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        

        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
    
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.num_feat)
            res = res + z
            
        if (self.output):
            res = self.output(res)
        return res;
    


















    
    
    
    
    
    
    
    
    