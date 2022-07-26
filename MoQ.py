#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
 


class Manager(nn.Module):
    
    def __init__(self, seq_len, in_channels, num_experts, horizon):
        """
        Parameters:
            - seq_len: the length of input time series, referring to the most recent 'p' observations.
            - in_channels: the number of features of the input (in_channels = 1 for univariate time series).
            - num_experts: the number of experts.
            - horizon: the number of forecasting horizons.
        
        """
        super(Manager, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.horizon = horizon
        self.fc = nn.Linear(seq_len*in_channels, horizon * num_experts).apply(weight_init)
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        """
        Input:
            - x: the input of Manager, (batch_size, seq_len, in_channels). 
            
        Output:
            - experts_score: the softmax score of experts, (batch_size, horizon, num_experts). 
        
        """
        x = x.contiguous().view(-1, self.seq_len*self.in_channels)
        out = self.fc(x)
        out = torch.reshape(out, (-1, self.horizon, self.num_experts))    # (batch_size, horizon, num_experts) 
        experts_score = self.softmax(out)
        return experts_score




class MoQ(nn.Module):

    def __init__(self, manager_model, num_experts):
        """
        Parameters:
            - manager_model: the created model of manager. 
                             E.g.,  
                             
                             manager = Manager(len_recent_obv, input_channels, num_experts, horizon)
                             model = MoQ(manager, num_experts)
                             
            - num_experts: the number of experts.
            
        """
        super(MoQ, self).__init__()
        self.manager = manager_model
        self.num_experts = num_experts
        

    def forward(self, x, preds_expert):
        """
        Input:
            - x: the recent observations of time series, (batch_size, len_recent_obv, in_channels).
            
            - preds_expert: the list of predictions made by experts [E_0.5, E_0.7, E_0.8, E_0.9]
                            [(batch_size, horizon, out_channels), ..., (batch_size, horizon, out_channels)].
        
        Output:
            - final_pred: the fused predictions, (batch_size, horizon, out_channels).
            - experts_score: the expert scores, (batch_size, horizon, num_experts).
        
        """
        experts_score = self.manager(x)
        weighted_pred = [preds_expert[i] * experts_score[:, :, [i]] for i in range(self.num_experts)]
        final_pred = torch.stack(weighted_pred).sum(dim=0)
        return final_pred, experts_score  
    
  
    
































