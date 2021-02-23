# Evaluation script
# This script provides one way to run LassoNet and competing feature selection methods
# on a given dataset, and to evaluate the performance of the selected features
# using downstream supervised learners.

###############################
#####       Imports      ######
###############################

from collections import defaultdict
import matplotlib.pyplot as plt
from os.path import join
from pathlib import Path
import torch
import numpy as np
import operator
import pickle
from sklearn.model_selection import train_test_split

from data_utils import load_mice, load_epileptic, load_coil, load_isolet, load_activity, \
        load_mnist, load_fashion, load_mnist_two_digits

from lassonet.lassonet_trainer import lassonet_trainer

figure_dir = 'figures'


###############################
#####    Other methods    #####
###############################
    
# Fisher Score
from skfeature.function.similarity_based import fisher_score
def fisher(train, test, K):
    score = fisher_score.fisher_score(train[0], train[1])
    indices = fisher_score.feature_ranking(score)[:K]
    return indices


# HSIC
from pyHSICLasso import HSICLasso
def hsic(train, test, K):
    hsic_lasso = HSICLasso()
    hsic_lasso.input(train[0],train[1])
    hsic_lasso.classification(K, n_jobs = -1)
    indices = hsic_lasso.get_index()
    return indices


# PFA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def pfa_selector(A, k, debug = False):
  class PFA(object):
      def __init__(self, n_features, q=0.5):
          self.q = q
          self.n_features = n_features

      def fit(self, X):
          if not self.q:
              self.q = X.shape[1]

          sc = StandardScaler()
          X = sc.fit_transform(X)

          pca = PCA(n_components=self.q).fit(X)
          self.n_components_ = pca.n_components_
          A_q = pca.components_.T
          
          print("nfeatures: {}".format(self.n_features))
          kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
          clusters = kmeans.predict(A_q)
          cluster_centers = kmeans.cluster_centers_

          self.indices_ = [] 
          for cluster_idx in range(self.n_features):
            indices_in_cluster = np.where(clusters==cluster_idx)[0]
            if len(indices_in_cluster)==0:
                continue
            points_in_cluster = A_q[indices_in_cluster, :]
            centroid = cluster_centers[cluster_idx]
            distances = np.linalg.norm(points_in_cluster - centroid, axis=1)
            optimal_index = indices_in_cluster[np.argmin(distances)]
            self.indices_.append(optimal_index) 
  
  pfa = PFA(n_features = k)
  pfa.fit(A)
  if debug:
    print('Performed PFW with q=', pfa.n_components_)
  column_indices = pfa.indices_
  return column_indices

def pfa_transform(A, B, k, debug = False):
    indices = pfa_selector(A[0], k, debug)
    return indices


# CIFE
from skfeature.function.information_theoretical_based import CIFE 
# see http://featureselection.asu.edu/html/skfeature.function.information_theoretical_based.CIFE.html
def CIFE_ours(train,test, K):
    idx,_,_ = CIFE.cife(train[0], train[1], n_selected_features=K)
    return idx[0:K]


# Trace Ratio
from skfeature.function.similarity_based import trace_ratio
def trace(train, test, K):
    idx, _, _ = trace_ratio.trace_ratio(train[0], train[1], K, style='fisher')
    return idx


from collections import defaultdict
def alternative_methods(name, train, test, algs, feature_sizes):
    # load any previous results
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    try:
        with open(join(figure_dir, name+'-indices.pkl'), 'rb') as f:
            indices = pickle.load(f)
    except FileNotFoundError:
            indices = defaultdict(dict)
    
    for alg in algs:
        print("\n\n Running %s"%alg.__name__)

        for k in feature_sizes:
            print('k = {}, algorithm = {}'.format(k, alg.__name__))
            indices_alg_k = alg((train[0], train[1]), (test[0], test[1]), k)
            indices[alg.__name__][k] = indices_alg_k
            
    # save the results
    pickle.dump(indices, open(join(figure_dir, name+'-indices.pkl'), 'wb'))
    
    return indices



#########################################
####  Downstream Supervised Learner  ####
####  1-hidden layer NN              ####
#########################################

import torch
from lassonet.models import DeepLasso12

def decoder(train, test, utils):
    device = utils['device']
    X_train, X_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    X_test, y_test = test
    

    # retrieve hyperparameters
    LR = utils['LR']

    # format X data
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    # Loss function
    crit = utils['criterion']
    hidden_dims = utils['hidden_dims']
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

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr = LR)

    def closure():
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        return loss

    model.train()

    epochs_since_best_obj = 0
    best_obj = np.inf
    for epoch in range(utils['n_epochs']):
        obj = optimizer.step(closure)
        with torch.no_grad():
            model.eval()
            obj = criterion(model(X_val_t),y_val_t).item()
        if obj < best_obj:
            best_obj = obj
            epochs_since_best_obj = 0
        if (utils['patience'] > 0) and (epochs_since_best_obj == utils['patience']):
            break
        epochs_since_best_obj += 1
    utils['train_loss'] = obj

    # record test error
    with torch.no_grad():
        model.eval()
        scores = model(X_test_t).detach()
        if crit=='CrossEntropy':
            pred = torch.argmax(scores, 1)
            accuracy = (pred == y_test_t).sum().item()/y_test_t.shape[0]
        elif crit=='MSE':
            accuracy = ((scores - y_test_t)**2).mean().item()

    utils['test_accuracy'] = accuracy
    del X_train_t, X_test_t, y_train_t, y_test_t, model
    return utils



#########################################
###  Evaluate on downstream learners ####
#########################################

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
def eval_methods(name, train, test, algs, feature_sizes,decoder_utils):
    '''
    'train' and 'test' are tuples of the form (X_train, y_train)
    'indices' give the indices of the selected subset of features
    '''
    X_train, y_train = train
    X_test, y_test = test
    n_clusters = len(np.unique(y_train))
    
    # load the indices giving the selected features
    try:
        indices = pickle.load(open(join(figure_dir, name+'-indices.pkl'), 'rb'))
    except FileNotFoundError:
        raise RuntimeError('No such file %s'%(join(figure_dir, name+'-indices.pkl')))

        
    # load any previously computed results
    try:
        results = pickle.load(open(join(figure_dir, name+'-results.pkl'), 'rb'))
    except FileNotFoundError:
        results = defaultdict(dict)
        
        
    def find_nearest(value, array):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]
    
    # run the reconstruction algorithms
    for alg in algs:
        indices_alg = indices[alg.__name__]
        print("\n\n Evaluating {} algorithm".format(alg.__name__))
        for feature_size in feature_sizes:
            print("decoder - size {}".format(feature_size))
            # find the closest computed size to feature_size
            # this is helpful when the lassonet path doesn't contain the exact feature_size requested
            _, k = find_nearest(feature_size, list(indices_alg.keys()))
            indices_alg_k = indices_alg[k]
            X_train_subset = X_train[:,indices_alg_k]
            X_test_subset = X_test[:,indices_alg_k]
    
            # Extra Trees Classifier
            clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1)
            clf.fit(X_train_subset, y_train)
            DTacc = float(clf.score(X_test_subset, y_test))

            # Decoder NN
            acc = decoder((X_train_subset, y_train), (X_test_subset, y_test), decoder_utils)['test_accuracy']
            
            # Logistic Regression
#             clf = LogisticRegression(n_jobs=-1, solver = 'sag')
#             clf.fit(X_train_subset, y_train)
#             LogisticAcc = float(clf.score(X_test_subset, y_test))
            
            # save results
            if 'decoder_accuracy' not in results[alg.__name__]:
                results[alg.__name__]['decoder_accuracy'] = {}
            results[alg.__name__]['decoder_accuracy'][k] = acc
            
            if 'tree_accuracy' not in results[alg.__name__]:
                results[alg.__name__]['tree_accuracy'] = {}
            results[alg.__name__]['tree_accuracy'][k] = DTacc

    pickle.dump(results, open(join(figure_dir, name+'-results.pkl'), 'wb'))
    return


if __name__=='__main__':
    # the main evaluation loop
    name = 'fashion'
    train, test = load_fashion()
    print("Using {} dataset".format(name))

    utils = {
        'M':30,
        'LR':1e-3,
        'hidden_dims':[500],
        'n_epochs':3000,
        'criterion':'CrossEntropy',
        'patience':50,
        'device':torch.device("cuda:2"),
        'figure_dir':'figures'
    }
    
    # run LassoNet and save the results to a file
    lassonet_trainer(name, train, test, utils)
    
    # run the other feature selection methods and save the results to a file
    feature_sizes = [25,50,75,100,125]
    alternative_methods(name,train,test,[fisher,hsic,pfa_transform],feature_sizes)
    
    # run the downstream supervised learners and save the results to a file
    eval_methods(name,train,test,[lassonet_trainer, fisher, hsic, pfa_transform],feature_sizes,utils)
    
    # print the results
    indices = pickle.load(open(join(figure_dir, name + '-indices.pkl'), 'rb'))
    results = pickle.load(open(join(figure_dir, name + '-results.pkl'), 'rb'))
    algs = {'lassonet_trainer': 'LassoNet', 'fisher':'Fisher', 'pfa_transform':'PFA',
            'hsic':'HSIC-LASSO'}
    res_by_k = {}
    for alg in algs:
        for k,v in results[alg]['tree_accuracy'].items():
            if k not in res_by_k:
                res_by_k[k] = dict()
            res_by_k[k][alg] = v

    for k,v in sorted(res_by_k.items(), key = operator.itemgetter(0)):
        print("k = {} - decoder accuracy = {}\n".format(k,v))