#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data
import os
import pickle
import numpy as np


class Quantile_Loss(nn.Module):
    
    def __init__(self):
        super(Quantile_Loss, self).__init__()
        
    def forward(self, y_pred, y_true, q):
        """
        y_pred: tensor of predictions, (batch_size, horizon, out_channels).
        y_true: tensor of ground-truth, (batch_size, horizon, out_channels).
        q: quantile index.
        
        """
        zeros = torch.zeros_like(y_true) 
        loss = q*torch.max(y_true-y_pred, zeros) + (1-q)*torch.max(y_pred-y_true, zeros)
        return torch.mean(loss)



def Penalization_Mask(y_true, u, num_expert=4, device=None):
    """
    Creating the penalization mask [0.25, 0.5, 0.75, 1.0] for the experts [E_0.5, E_0.7, E_0.8, E_0.9].
    
    Parameters:
        - y_true: tensor of ground-truth, (batch_size, horizon, out_channels).
        
        - u: the number of penalized samples (10% of all samples used in the paper).
             E.g., u = int(np.ceil(batch_size/10)) 
             
        - num_expert: the number of experts.
            
        - device: e.g., device = torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        
    Output:
        - expert_mask: a list of penalization masks, len(expert_mask) = num_expert.
                       [(batch_size, horizon, out_channels), ..., batch_size, horizon, out_channels].
                       
    """
    y_sum = torch.sum(torch.sum(y_true, dim=1), dim=1)
    val, idx = torch.topk(y_sum, u)
    expert_mask = []
    
    for i in range(1, num_expert+1):
       mask = torch.ones_like(y_true)
       mask[idx] = mask[idx] * i / num_expert
       
       if device != None:
           mask.to(device)
       
       expert_mask.append(mask)
       
    return expert_mask



def Penalization_GaussianNoise(y_true, u, expert_quantiles=[0.5, 0.7, 0.8, 0.9], alpha=2, device=None):
    """
    Creating the aggressiveness-related Gaussian noise for the provided expert_quantiles.
    
    Parameters:
        - y_true: tensor of ground-truth, (batch_size, horizon, out_channels).
        
        - u: the number of penalized samples (10% of all samples used in the paper).
             E.g., u = int(np.ceil(batch_size/10)) 
             
        - expert_quantiles: the quantile index used to train experts.
             E.g., preds_expert = [Pred_E_0.5, Pred_E_0.7, Pred_E_0.8, Pred_E_0.9]
                   expert_quantiles = [0.5, 0.7, 0.8, 0.9]
                   
        - alpha: the paprameter controlling the variance of Gaussian noise.
        
        - device: e.g., device = torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        
    Output:
        - expert_noise: a list of aggressiveness-based Gaussian noises, len(expert_noise) = len(expert_quantiles).      
                       [(batch_size, horizon, out_channels), ..., batch_size, horizon, out_channels].

    """
    y_sum = torch.sum(torch.sum(y_true, dim=1), dim=1)
    val, idx = torch.topk(y_sum, u)
    expert_noise = []
    
    for q in expert_quantiles:
       mean = torch.zeros_like(y_true)
       var_scale = alpha * (1/q - 1)
       var = torch.ones_like(y_true) * var_scale
       std = torch.sqrt(var)
       GaussianSamples = torch.normal(mean=mean, std=std)
       noise = torch.zeros_like(y_true)
       noise[idx] = GaussianSamples[idx]
       
       if device != None:
           noise.to(device)
       
       expert_noise.append(noise)
       
    return expert_noise



def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save(model, prefix_file, architecture=''):
    torch.save(
        model.state_dict(),
        prefix_file + architecture + '.pth'
    ) 


def load(model, prefix_file, architecture='', cuda = False, gpu = 0):
    if cuda:
        model.load_state_dict(torch.load(
            prefix_file + architecture + '.pth',
            map_location=lambda storage, loc: storage.cuda(gpu)
        ))
        
    else:
        model.load_state_dict(torch.load(
            prefix_file + architecture + '.pth',
            map_location=lambda storage, loc: storage
        ))


def read_variable(filename, file_dir):
    if '.pckl' not in filename:
        filename += '.pckl'
    f = open(file_dir+filename, 'rb')
    var = pickle.load(f)
    f.close()
    return var


class RegressionDataset(torch.utils.data.Dataset):   
    
    def __init__(self, dataset, horizons=None, channel_first=True , forecasting_horizons=4):
        """
    
        Parameters
        ----------
        dataset : Ndarray of shape (dim_N, dim_C, dim_T) 
            dim_N: the number of objects (cells).
            dim_T: the time legnth.
            dim_C: the number of channels/features.
            
        horizons : Ndarray of shape (dim_N, dim_C, dim_H), optional
            dim_N: the number of objects (cells).
            dim_H: the forecasting horizons.
            dim_C: the number of predicted channels/features.
            
            The ground truth of this regression dataset. If this array is not given, 
            the ground truth array will be created based on dataset and the forecasting
            horizons. The default is None.
            
        
        channel_first : boolean, optional
            If true, the dataset shape is (dim_N, dim_C, dim_T), otherwwise (dim_N, dim_T, dim_C) 
        
            
        forecasting_horizons : Integer, optional
            The number of predicted time steps. The default is 4.

        Returns
        -------
        None.

        """
        if not horizons:
            self.forecasting_horizons = forecasting_horizons
            
            if channel_first:
                self.dataset = dataset[:, :, :-1*self.forecasting_horizons]
                self.horizons = dataset[:, :, -1*self.forecasting_horizons:]
                
            else:
                self.dataset = dataset[:, :-1*self.forecasting_horizons, :]
                self.horizons = dataset[:, -1*self.forecasting_horizons:, :]
            
        else:
            self.dataset = dataset
            self.horizons = horizons
            
        
    def __len__(self):
        return np.shape(self.dataset)[0]
    
    def __getitem__(self, index):
        return self.dataset[index], self.horizons[index]



