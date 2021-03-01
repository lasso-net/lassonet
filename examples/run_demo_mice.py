#!/usr/bin/env python
# coding: utf-8

# ## Lassonet Demo Notebook - PyTorch
# 
# This notebook illustrates the Lassonet method for feature selection on a classification task.
# We will run Lassonet over [the Mice Dataset](https://archive.ics.uci.edu/ml/datasets/Mice%20Protein%20Expression). This dataset consists of protein expression levels measured in the cortex of normal and trisomic mice who had been exposed to different experimental conditions. Each feature is the expression level of one protein.

# First we import a few necessary packages

# In[1]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lassonet.lassonet_trainer import lassonet_trainer


# Then, loading the Mice dataset

# In[2]:


def load_mice(one_hot = False):
    filling_value = -100000
    X = np.genfromtxt('Data_Cortex_Nuclear.csv', delimiter = ',', skip_header = 1, usecols = range(1, 78), filling_values = filling_value, encoding = 'UTF-8')
    classes = np.genfromtxt('Data_Cortex_Nuclear.csv', delimiter = ',', skip_header = 1, usecols = range(78, 81), dtype = None, encoding = 'UTF-8')

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if val == filling_value:
                X[i, j] = np.mean([X[k, j] for k in range(classes.shape[0]) if np.all(classes[i] == classes[k])])

    DY = np.zeros((classes.shape[0]), dtype = np.uint8)
    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ['Control', 'Memantine', 'C/S'])):
            DY[i] += (2 ** j) * (val == label)

    Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
    for idx, val in enumerate(DY):
        Y[idx, val] = 1

    X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    DY = DY[indices]
    classes = classes[indices]
    
    if not one_hot:
        Y = DY
        
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # print(X.shape, Y.shape)
    print("data shape : {}".format(X.shape))
    
    return (X[: X.shape[0] * 4 // 5], Y[: X.shape[0] * 4 // 5]), (X[X.shape[0] * 4 // 5:], Y[X.shape[0] * 4 // 5: ])

name = 'mice'
train, test = load_mice()


# LassoNet natively convers the data into PyTorch Tensors, and is GPU-compatible. If the machine has a GPU, it will be used for training. Otherwise, it defaults to CPU.
# 
# LassoNet uses a training loop typical of PyTorch. We begin by training an initial fully dense, and use it as warm start over the entire regularization path. The entire training loop takes less than 5 min on a Tesla K80 GPU.

# In[4]:


lassonet_results = lassonet_trainer(name, train, test)


# The classification accuracy is plotted below:

# In[10]:


nselected = lassonet_results['nselected']
accuracy = lassonet_results['test_accuracy']

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (9,6))
plt.grid(True)
plt.plot(nselected, accuracy, 'o-')
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")
plt.title("Classification accuracy")
plt.savefig('accuracy.png')


# To view the selected features, we can load the 'indices' list from the trainer.

# In[16]:


import pickle
indices = pickle.load(open(name+'-indices.pkl','rb'))
# print(indices['lassonet'])

# To access snapshots of the model along the regularization path, we can call
# lassonet_results['snapshots']. More details in 'lassonet_trainer.py'
