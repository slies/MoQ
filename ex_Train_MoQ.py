#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import argparse
import utils, LSTNet, MoQ
import metrics


def load_expert():
    PATH_root = os.getcwd()
    PATH_data = PATH_root + '/Data/'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 24*14 - 4
    input_channels = 1
    
    # Load Expert

    architecture = '_LSTNet'  
    PATH_expert = PATH_root + '/Expert/'
    config = utils.read_variable('params_expert', PATH_data)
    expert = []
    
    for q in [0.5, 0.7, 0.8, 0.9]:
        hidSkip = 5
        skip = 24
        hidCNN = config[q]['hidCNN']
        hidRNN = config[q]['hidRNN']
        dropout = config[q]['dropout']
    
        
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
        model_expert = LSTNet.LSTNet(args)
        model_expert = model_expert.to(device)
        utils.load(model_expert, PATH_expert+'expert_'+str(q), architecture=architecture)  
        model_expert.eval()
        expert.append(model_expert)
    
    num_expert = len(expert)
    
    return expert, num_expert



def train(penalization):
    PATH_root = os.getcwd()
    PATH_data = PATH_root + '/Data/'
    PATH_model = PATH_root + '/MoQ/'
    utils.check_path(PATH_model)
    architecture = 'MoQ'
    
    train = utils.read_variable('PM25_train', PATH_data)
    valid = utils.read_variable('PM25_valid', PATH_data)    
    train = torch.from_numpy(train).float()
    valid = torch.from_numpy(valid).float()
    
    combined_train = torch.cat([train, valid])
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.L1Loss()
    
    input_channels = 1
    predict_idx = [0]    
    horizon = 4
    
    expert, num_expert = load_expert()
    
    config = utils.read_variable('params_MoQ', PATH_data)
    Epochs = config[penalization]['Epochs']
    batch_size = config[penalization]['batch_size']
    lr = config[penalization]['lr']
    u = config[penalization]['u']
    recent_obv = config[penalization]['recent_obv']

    manager = MoQ.Manager(recent_obv, input_channels, num_experts=num_expert, horizon=horizon)
    model = MoQ.MoQ(manager, num_experts=num_expert)
    model.to(device)
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

            Pred_expert = []
                                
            with torch.no_grad():
            
                for model_expert in expert:
                    data_x = train_x*torch.Tensor([1.]).to(device)
                    y_pred = []
                   
                    for h in range(horizon):
                        oneStep_pred = model_expert(data_x)
                        oneStep_pred = torch.unsqueeze(oneStep_pred, dim=1)
                        y_pred.append(oneStep_pred)
                        data_x = torch.cat((data_x, oneStep_pred), dim=1)      
                        data_x = data_x[:,1:,:]
                    
                    y_pred = torch.cat(y_pred, dim=1) 
                    Pred_expert.append(y_pred)
            
            if train_y.shape[0] <= u:
                Pred_expert_penalized = Pred_expert
            else:
                if penalization == 'mask':
                    penalization_tensor = utils.Penalization_Mask(train_y, u, num_expert, device)
                    Pred_expert_penalized = [pred*noise for pred, noise in zip(Pred_expert, penalization_tensor)]
                else:
                    alpha = 4
                    penalization_tensor = utils.Penalization_GaussianNoise(train_y, u, alpha=alpha, device=device)
                    Pred_expert_penalized = [pred+noise for pred, noise in zip(Pred_expert, penalization_tensor)]
            
            y_pred, expert_score = model(train_x[:, -recent_obv:, :], Pred_expert_penalized)
            
            loss = criterion(y_pred, train_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            if idx%20 == 0:      
                print('Iteration %d, MAE %.5f ' %(idx, loss.item()))    
            idx += 1
                
        print('\nEpoch %d: finished' %(i+1))
            
    utils.save(model, PATH_model+'MoQ_'+penalization, architecture=architecture)
        
        

def test(penalization):
    PATH_root = os.getcwd()
    PATH_data = PATH_root + '/Data/'
    PATH_model = PATH_root + '/MoQ/'
    utils.check_path(PATH_model)
    architecture = 'MoQ'
    
    test = utils.read_variable('PM25_test', PATH_data)
    test = torch.from_numpy(test).float()
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_set = utils.RegressionDataset(test, channel_first=False)
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=30)    
    
    input_channels = 1
    predict_idx = [0]    
    horizon = 4
    
    expert, num_expert = load_expert()
    
    config = utils.read_variable('params_MoQ', PATH_data)
    recent_obv = config[penalization]['recent_obv']
    
    manager = MoQ.Manager(recent_obv, input_channels, num_experts=num_expert, horizon=horizon)
    model = MoQ.MoQ(manager, num_experts=num_expert)
    model.to(device)
    utils.load(model, PATH_model+'MoQ_'+penalization, architecture=architecture)  
    model.eval()

    MAE = []
    MSE = []
    Pred_MoQ = []
    Ground_truth = []
    
    criterion = nn.L1Loss()    
    criterion_mse = nn.MSELoss()
    
    with torch.no_grad():
        
        for test_data, test_horizon in test_generator:        
            test_data = test_data.to(device)
            test_horizon = test_horizon.to(device)
            
            test_x = test_data[:,:,predict_idx]
            test_y = test_horizon[:,:,predict_idx]
                    
            Pred_expert = []
            
            for model_expert in expert:
                data_x = test_x*torch.Tensor([1.]).to(device)
                y_pred = []
               
                for h in range(horizon):
                    oneStep_pred = model_expert(data_x)
                    oneStep_pred = torch.unsqueeze(oneStep_pred, dim=1)
                    y_pred.append(oneStep_pred)
                    data_x = torch.cat((data_x, oneStep_pred), dim=1)      
                    data_x = data_x[:,1:,:]
                
                y_pred = torch.cat(y_pred, dim=1) 
                Pred_expert.append(y_pred)
                
            y_pred, _ = model(test_x[:, -recent_obv:, :], Pred_expert)
            
            loss = criterion(y_pred, test_y)
            loss_mse = criterion_mse(y_pred, test_y)

            MAE.append(loss.item())
            MSE.append(loss_mse.item())
            Pred_MoQ.append(y_pred.cpu().detach().numpy())
            Ground_truth.append(test_y.cpu().detach().numpy())

    Ground_truth = np.concatenate(Ground_truth, axis=0)
    Pred_MoQ = np.concatenate(Pred_MoQ, axis=0)
    accuracy_avg, (_, sensitivity) = metrics.AVG_classification_accuracy(Ground_truth, Pred_MoQ, quantile=0.95)

    print('\nMoQ [' + penalization + '] \n')
    print('Test Set, MAE Loss ', np.mean(MAE))
    print('Test Set, MSE Loss ', np.mean(MSE))
    print('AVG Accuracy %.3f' % (accuracy_avg*100), '%', 
          'Sensitivity %.3f' % (sensitivity*100), '%')
    

       

if __name__ == '__main__':
    
    penalization = 'mask'  # 'mask' or 'Gaussian']
    train(penalization)
        
    penalization = 'Gaussian' 
    train(penalization)
    
    #%%
    
    penalization = 'mask'  
    test(penalization)
    
    penalization = 'Gaussian' 
    test(penalization)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    