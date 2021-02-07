#!/usr/bin/python
#-*- coding: utf-8 -*- 
"""
metric.py
    python function for evaluating x and y 
    The x and y must have the same shape, and the first dimension must be a batch size (sample).
    Aim to push into bdpy
"""

import numpy as np
from numpy.matlib import repmat
from tqdm import tqdm

def squarederror(x,y):
    """Calculate squared error

    Args:
        x (np.array): target (decoded/recon) values [batch_size, *shape]
        y (np.array): soruce (true) values [batch_size, *shape]


    Returns:
        np.array: (x - y) ** 2 
    """
    diff_squared = (x - y)**2
    return np.sqrt(diff_squared)

def corr(x, y, var='col'):
    """[summary]

    Args:
        x (np.array): target (decoded/recon) values [batch_size, *shape]
        y (np.array): soruce (true) values [batch_size, *shape]
        var (str, optional): Specifying whther rows or colmuns represent variables. Defaults to 'col'.

    Return:
        r
         Correlation coefficient of each columns or rows
    """
    
    batch_size = x.shape[0]
    stim_shape = x.shape[1:]
    x_flat = x.reshape(batch_size, -1)
    y_flat = y.reshape(batch_size, -1)
    # Normalize x and y to row-var format
    if var =='row':
        if x_flat.shape[1] ==1:
            x_flat = x_flat.T
        if y_flat.shape[1] == 1:
            y_flat = y_flat.T
    elif var == 'col':
        x_flat = x_flat.T
        y_flat = y_flat.T
    else:
        raise ValueError('Unknonn var parameter specified')

    # Match size of x and y
    if x_flat.shape[0] ==1 and y_flat.shape[0] != 1:
        x_flat = repmat(x_flat, y_flat.shape[0], 1)
    elif x_flat.shape[0] !=1 and y_flat.shape[0] == 1:
        y_flat = repmat(y_flat, x_flat.shape[0], 1)

    # Check size of normalized x and y
    if x_flat.shape != y_flat.shape:
        raise TypeError('Input matrixes size mismatch')

    #Get num variables
    nvar = x_flat.shape[0]
    nunits = x_flat.shape[1]

    #Get Correlation for each row for avoding memory error
    r = np.array([np.corrcoef(x_flat[i].flatten(), y_flat[i].flatten())[0,1] for i in tqdm(range(nvar))])
    
    if var == 'col':
        return r.reshape(stim_shape)
    else:
        return r

def pairwise_identification(x,y):
    """[summary]

    Args:
        x (np.array): target (decoded/recon) values [batch_size, *shape]
        y (np.array): soruce (true) values [batch_size, *shape]
    """

    #make similarity matrix
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    simmat = corrmat(x_flat,y_flat)

    num_pred = simmat.shape[0]
    labels = np.arange(num_pred)
    
    correct_rate = []
    for i, label in enumerate(labels):
        pred_feat = simmat[i]
        correct_feat = pred_feat[label]
        pred_num = len(pred_feat) - 1
        correct_rate.append((pred_num - np.sum(pred_feat > correct_feat)) / pred_num)
    return np.array(correct_rate)

def corrmat(x,y, var='row'):
    """[summary]

    Args:
        x (np.array): target (decoded/recon) values [batch_size, *shape]
        y (np.array): soruce (true) values [batch_size, *shape]

    Returns:
        similarity matrix
    """

    if var == 'col':
        x = x.T
        y = y.T
    elif var == 'row':
        pass
    else:
        raise ValueError('Unknown var parameter specified')

    nobs = x.shape[1]

    # Subtract mean(a, axis=1) from a
    submean = lambda a: a - np.matrix(np.mean(a, axis=1)).T
    
    cmat = (np.dot(submean(x), submean(y).T) / (nobs - 1)) / np.dot(np.matrix(np.std(x, axis=1, ddof=1)).T, np.matrix(np.std(y, axis=1, ddof=1)))
    
    return np.array(cmat)
    

if __name__ == '__main__':
    rand_img = np.random.rand(1, 224,224, 3)

    print(rms(rand_img, rand_img))