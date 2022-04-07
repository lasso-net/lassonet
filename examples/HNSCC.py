#!/usr/bin/env python
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, scale
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import csv, os
import pickle
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
import pandas as pd

scriptpath = '../'
#scriptpath = "/Users/xuelinyang/Documents/lassonet/"
sys.path.append(os.path.abspath(scriptpath))

from lassonet.interfaces import LassoNetRegressor, LassoNetCoxRegressor
from lassonet.utils import plot_cox_loss

# download dataset
import gdown
url = 'https://docs.google.com/uc?export=download&id=1e17cHysfhVt-w-o2w4IGCQYoLpetvuEi'
output = 'x_glmnet_2.csv'
gdown.download(url, output)

url = 'https://docs.google.com/uc?export=download&id=1OyxX33rrG2ERrpATH61o6e49idypEGdp'
output = 'y_glmnet_2.csv'
gdown.download(url, output)


X = pd.read_csv('x_glmnet_2.csv')
y = pd.read_csv('y_glmnet_2.csv')
X = X.to_numpy()
y = y.to_numpy().astype(float)


# define hyper params
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
lambda_start = 1e-4
hidden_dim = 1
path_multiplier = 1.1
eps_start = 0.1

model = LassoNetCoxRegressor(
    hidden_dims=(hidden_dim,),
    eps_start=eps_start,
    lambda_start=lambda_start,
    path_multiplier=path_multiplier,
    verbose=True,
    batch_size=None,
)

# save results
unique_name = "_train_cox_linear"
path = model.path(X_train, y_train)
dir_path = "HNSCC_results_4_5_randstate_"  + str(random_state)
if not os.path.exists(dir_path):
	os.makedirs(dir_path)
prefix = dir_path + "/hidden_" + str(hidden_dim)+ "_eps_"+ str(eps_start) +"_lambd_" + str(lambda_start)+"_pm_"+ str(path_multiplier) + "_bs_None" + unique_name + "_loss"

plot_cox_loss(model, path, X_test, y_test)
#plot_cox_loss(model, path, X_test, y_test, is_twin=True, ori_loss_type='cox')
plt.savefig(prefix+".png")
