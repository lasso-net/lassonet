import pickle
from pathlib import Path
from os.path import join
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np

figure_dir = 'figures'

###############################
######     Lassonet      ######
###############################

import torch
from deeplasso.models import DeepLasso12
def lassonet(name, train, test, utils):

    device = utils['device']
    patience = utils.get('patience', 10)
    
    X_train, X_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    X_test, y_test = test
    
    # format X data
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    # Loss function
    crit = utils['criterion']
    hidden_dims = utils.get('hidden_dims',100)
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


    # path training
    alpha_max_lasso = (X_train.T.dot(y_train)/X_train.shape[0]).max()
    alpha_min = alpha_max_lasso * utils['lambda_min_multiplier']
    
    optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum = 0.9)
    alphas = [alpha_min]
    
    coefs = [model.skip_connections[0].weight.detach().cpu().numpy().max(axis=0)]
    nselected = [nfeats]
    train_accs = [train_acc]
    test_accs = [test_acc]
    indices = {nfeats:np.arange(nfeats)}
    while nselected[-1] != 0:
        best_obj = np.inf
        epochs_since_best_obj = 0
        model.train()
        for epoch in range(utils.get('n_epochs',400)):
            loss = optimizer.step(closure)
            model.prox(lambda_ = alphas[-1] * LR, M = utils['M'])
            with torch.no_grad():
                model.eval()
                obj = (criterion(model(X_val_t),y_val_t) + alphas[-1] * model.regularization()).item()
            if obj < best_obj:
                best_obj = obj
                epochs_since_best_obj = 0
            if (patience > 0) and (epochs_since_best_obj == patience):
                break
            epochs_since_best_obj += 1

        # record number of selected features
        coef = model.skip_connections[0].weight.detach().cpu().numpy().max(axis=0)
        coefs.append(coef)
        nsel = (coef!=0).sum()
        if nsel != nselected[-1] and nsel > 0:
            indices[nsel] = np.where(coef != 0)[0]
            print("selected {} features".format(nsel))
        nselected.append(nsel)

    
        # record test error
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
                
        alphas.append(utils['path_multiplier']*alphas[-1])

    utils['coefs'] = np.array(coefs)
    utils['nselected'] = np.array(nselected)
    utils['train_accuracy'] = np.array(train_accs)
    utils['test_accuracy'] = np.array(test_accs)
    utils['alphas'] = np.array(alphas)
    
    
    # save Lassonet results
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