# This file provides the main training loop for LassoNet

import pickle
from pathlib import Path
from os.path import join
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np


###############################
######     Lassonet      ######
###############################

import torch
from .models import DeepLasso12

def lassonet_trainer(name, train, test, utils = {}):
    """
    Main training loop for LassoNet
    
    Inputs:
        -- name: the name of the dataset. Is used to create a folder to save the results after training. See 'figure_dir' argument in utils below
        -- train: a tuple (X_train, y_train) containing the training data
        -- test : a tuple (X_test, y_test) containing the test data
        -- utils [optional]: a dictionary containing training hyperparameters:
              - 'M': The hierarchy parameter. Default: 10
              - 'device': the device on which to train the model using PyTorch. Default: GPU 0 if available, CPU otherwise.
              - 'criterion': 'CrossEntropy' for classification problems, 'MSE' for regression problems. Default: 'CrossEntropy'
              - 'LR' : learning rate. Default: 1e-3
              - 'hidden_dims' : list containing the size of all hidden layers. Default: [100]
              - 'n_epochs_init': Maximum number of epochs for training the initial dense model. This is an upper-bound on the      effective number of epochs, since the model uses early stopping. Default: 3000
              - 'n_epochs': Maximum number of epochs to run for the sparse training. Default: 400
              - 'path_multiplier': Multiplicative factor [also denoted "1 + epsilon" in the paper] to increase the penalty parameter over the path. Default: 1.05
              - 'figure_dir': Directory to store the results. The saved results will be files of the form 'figure_dir/name+xxx'
              - 'patience': the number of epochs to wait without improvement during early stopping. Default: 10
              
    
    Returns:
        -- utils: a dictionary containing the above hyperparameters with 4 additional keys:
              - 'coefs': array of shape (n_lambda, n_features) giving the residual coefficients
              - 'nselected': list of length (n_lambda) giving the number of selected features
              - 'train_accuracy': list of length n_lambda giving the accuracy [either classification accuracy or MSE] on the training dataset
              - 'test_accuracy': list of length n_lambda giving the accuracy [either classification accuracy or MSE] on the test dataset.
    utils['alphas'] = array of length n_lambda giving the values of the penalty lambda used.
    """

    device = utils.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    patience = utils.get('patience', 10)
    
    X_train, X_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    X_test, y_test = test
    
    # format X data
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Loss function
    crit = utils.get('criterion', 'CrossEntropy')
    hidden_dims = utils.get('hidden_dims',[100])
    nfeats = X_train.shape[1]
    if crit=='CrossEntropy':
        # format y data
        y_train_t = torch.LongTensor(y_train).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)
        y_test_t = torch.LongTensor(y_test).to(device)
        # define model
        nclasses = len(np.unique(y_test))
        model = DeepLasso12(nfeats, *hidden_dims, nclasses).to(device)
        # define loss function
        criterion = torch.nn.CrossEntropyLoss(reduction = 'mean')
    elif crit=='MSE':
        y_train_t = torch.FloatTensor(y_train).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        y_test_t = torch.FloatTensor(y_test).to(device)
        model = DeepLasso12(nfeats, *hidden_dims, 1).to(device)
        criterion = torch.nn.MSELoss(reduction='mean')
    else:
        raise Exception("'criterion' must be either 'CrossEntropy' or 'MSE'")
        
    def closure():
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        return loss

    
    # initial training
    # optimizer
    LR = utils.get('LR', 1e-3)
    optimizer = torch.optim.Adam(model.parameters(),lr = LR)
    model.train()
    epochs_since_best_obj = 0
    best_obj = np.inf
    for epoch in range(utils.get('n_epochs_init', 3000)):
        obj = optimizer.step(closure)
        with torch.no_grad():
            model.eval()
            obj = criterion(model(X_val_t),y_val_t).item()
        if obj < best_obj:
            best_obj = obj
            epochs_since_best_obj = 0
        if (patience > 0) and (epochs_since_best_obj == patience):
            break
        epochs_since_best_obj += 1
    
    with torch.no_grad():
        model.eval()
        train_scores = model(X_train_t).detach()
        test_scores = model(X_test_t).detach()
        if crit=='CrossEntropy':
            train_pred = torch.argmax(train_scores, 1)
            test_pred = torch.argmax(test_scores, 1)
            train_acc = (train_pred == y_train_t).sum().item()/y_train_t.shape[0]
            test_acc = (test_pred == y_test_t).sum().item()/y_test_t.shape[0]
        else:
            train_acc = ((train_scores - y_train_t)**2).mean().item()
            test_acc = ((test_scores - y_test_t)**2).mean().item()
        print("dense model achieved train/test accuracy:{:.3f}/{:.3f} after {} epochs".format(
                train_acc,test_acc,epoch))

   
    # save several metrics about the dense model
    coefs = [model.skip_connections[0].weight.detach().cpu().numpy().max(axis=0)]
    nselected = [nfeats]
    train_accs = [train_acc]
    test_accs = [test_acc]
    indices = {nfeats:np.arange(nfeats)}
    snapshots = [model.state_dict().copy()]

    # path training
    alpha_max_lasso = (X_train.T.dot(y_train)/X_train.shape[0]).max()
    alpha_min = alpha_max_lasso * utils.get('lambda_min_multiplier', 1e-2)
    optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum = 0.9)
    alphas = [alpha_min]

    while nselected[-1] != 0:
        best_obj = np.inf
        epochs_since_best_obj = 0
        model.train()
        for epoch in range(utils.get('n_epochs',400)):
            loss = optimizer.step(closure)
            model.prox(lambda_ = alphas[-1] * LR, M = utils.get('M',10))
            with torch.no_grad():
                model.eval()
                obj = (criterion(model(X_val_t),y_val_t) + alphas[-1] * model.regularization()).item()
            if obj < best_obj:
                best_obj = obj
                epochs_since_best_obj = 0
            if (patience > 0) and (epochs_since_best_obj == patience):
                break
            epochs_since_best_obj += 1

        # take snapshot of model
        snapshots.append(model.state_dict().copy())

        # record number of selected features
        coef = model.skip_connections[0].weight.detach().cpu().numpy().max(axis=0)
        coefs.append(coef)
        nsel = (coef!=0).sum()
        if nsel != nselected[-1] and nsel > 0:
            indices[nsel] = np.where(coef != 0)[0]
            print("selected {} features".format(nsel))
        nselected.append(nsel)

    
        # record test error
        if test is not None:
            with torch.no_grad():
                model.eval()
                train_scores = model(X_train_t).detach()
                test_scores = model(X_test_t).detach()
                if crit=='CrossEntropy':
                    train_pred = torch.argmax(train_scores, 1)
                    test_pred = torch.argmax(test_scores, 1)
                    train_acc =(train_pred == y_train_t).sum().item()/y_train_t.shape[0]
                    test_acc = (test_pred == y_test_t).sum().item()/y_test_t.shape[0]
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                else:
                    train_mse = ((train_scores - y_train_t)**2).mean().item()
                    test_mse = ((test_scores - y_test_t)**2).mean().item()
                    train_accs.append(train_mse)
                    test_accs.append(test_mse)
                utils['test_accuracy'] = np.array(test_accs)
                
        alphas.append(utils.get('path_multiplier',1.05)*alphas[-1])

    utils['coefs'] = np.array(coefs)
    utils['nselected'] = np.array(nselected)
    utils['train_accuracy'] = np.array(train_accs)
    utils['alphas'] = np.array(alphas)
    utils['snapshots'] = snapshots
    
    
    # save updated utils
    figure_dir = utils.get('figure_dir','.')
    filepath = join(figure_dir, name + '-lassonet-utils.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(utils, f)        
                      
    # save selected indices
    filepath = join(figure_dir, name+'-indices.pkl')
    try:
        ind = pickle.load(open(filepath, 'rb'))
    except FileNotFoundError:
        ind = defaultdict(dict)
    ind['lassonet'] = indices
    pickle.dump(ind,open(filepath, 'wb'))

    # free up memory
    del X_train_t, X_test_t, y_train_t,y_test_t, model
    return utils
