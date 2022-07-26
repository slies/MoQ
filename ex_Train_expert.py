#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import argparse
import utils, LSTNet


def train():
    PATH_root = os.getcwd()
    PATH_data = PATH_root + '/Data/'
    PATH_model = PATH_root + '/Expert/'
    utils.check_path(PATH_model)

    train = utils.read_variable('PM25_train', PATH_data)
    valid = utils.read_variable('PM25_valid', PATH_data)    
    train = torch.from_numpy(train).float()
    valid = torch.from_numpy(valid).float()
    
    combined_train = torch.cat([train, valid])
    
    architecture = '_LSTNet'    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion_quantile = utils.Quantile_Loss()
    
    seq_len = 24*14 - 4
    input_channels = 1
    predict_idx = [0]    
    horizon = 4
    hidSkip = 5
    skip = 24
    config = utils.read_variable('params_expert', PATH_data)
    
    
    for q in [0.9, 0.8, 0.7, 0.5]:
        hidCNN = config[q]['hidCNN']
        hidRNN = config[q]['hidRNN']
        dropout = config[q]['dropout']
        Epochs = config[q]['Epochs']
        batch_size = config[q]['batch_size']
        lr = config[q]['lr']
        
        parser = argparse.ArgumentParser(description='LSTNet')         
        parser.add_argument('--num_feat', type=int, default=input_channels)
        parser.add_argument('--hidCNN', type=int, default=hidCNN,
                            help='number of CNN hidden units')
        parser.add_argument('--hidRNN', type=int, default=hidRNN,
                            help='number of RNN hidden units')
        parser.add_argument('--hidSkip', type=int, default=hidSkip)
        parser.add_argument('--window', type=int, default=seq_len,
                            help='window size')
        parser.add_argument('--skip', type=int, default=skip)
        parser.add_argument('--CNN_kernel', type=int, default=4,
                            help='the kernel size of the CNN layers')
        parser.add_argument('--highway_window', type=int, default=16,
                            help='The window size of the highway component')
        parser.add_argument('--dropout', type=float, default=dropout,
                            help='dropout applied to layers (0 = no dropout)')
        parser.add_argument('--output_fun', type=str, default=None)
    
        args = parser.parse_args() 
        model = LSTNet.LSTNet(args)
        model = model.to(device)
        model.train()


        train_set = utils.RegressionDataset(combined_train, channel_first=False)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        
        for i in range(Epochs):
            idx = 0            
            
            for train_data, train_horizon in train_generator:        
                train_data = train_data.to(device)
                train_horizon = train_horizon.to(device)    
            
                train_x = train_data[:,:,predict_idx]
                train_y = train_horizon[:,:,predict_idx]
                y_pred = []
                                        
                for h in range(horizon):
                    oneStep_pred = model(train_x)
                    oneStep_pred = torch.unsqueeze(oneStep_pred, dim=1)
                    y_pred.append(oneStep_pred)
                    
                    train_x = torch.cat((train_x, train_y[:, [h], :]), dim=1)       # teacher forcing
                    train_x = train_x[:,1:,:]
            
                y_pred = torch.cat(y_pred, dim=1)    
                loss_quantile = criterion_quantile(y_pred, train_y, q)
                
                optimizer.zero_grad()
                loss_quantile.backward()
                optimizer.step()
                    
                if idx%20 == 0:      
                    print('Iteration %d, Qauntile Loss %.5f ' %(idx, loss_quantile))    
                idx += 1
                
            print('\nEpoch %d: finished' %(i+1))
            
        utils.save(model, PATH_model+'expert_'+str(q), architecture=architecture)
        
        

if __name__ == '__main__':
    
    train()
        
        
    
    
    
    
    
    
    