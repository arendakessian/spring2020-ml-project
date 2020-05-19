#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:18:17 2020

@author: kelseymarkey
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
import random
random.seed(22)

# Importing labels
with open('../data/train_labels.pckl', 'rb') as f:
    train_labels = pickle.load(f)

with open('../data/dev_labels.pckl', 'rb') as f:
    dev_labels = pickle.load(f)
    
    
def get_data(dataset, vectorizer):
    '''
    returns feature matrix for specified dataset and vectorizer
    @param dataset: string specifying dataset, "train","dev",etc
    @param vectorizer: string specifying vectorizer "binary","count",etc

    '''
    with open(f'../data/{dataset}_{vectorizer}_downsampled_data.pckl', 'rb') as f:
        return pickle.load(f)
    

# specify parameters and distributions to sample from
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 150, num = 8)]
max_features = ['log2']
max_depth = [int(x) for x in np.linspace(10, 600, num = 13)]
# max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

param_dist = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}


vectorizers = ['tfidf', 'count', 'binary']

for vectorizer in vectorizers:
    print('----- ', vectorizer, ' -----')
    train = get_data('train', vectorizer)
    dev = get_data('dev', vectorizer)

    rf = RandomForestClassifier(random_state=22)
        
    # run randomized search
    random_search = RandomizedSearchCV(estimator = rf, 
                                       param_distributions=param_dist,
                                       n_iter = 30,
                                       random_state=22,
                                       n_jobs = -1,
                                       verbose=2)
    
    random_search.fit(train, train_labels)
    
    rf_train = random_search.predict_proba(train)[:, 1]
    rf_dev = random_search.predict_proba(dev)[:, 1]
    
    rf_train_auc = roc_auc_score(train_labels, rf_train)
    rf_dev_auc = roc_auc_score(dev_labels, rf_dev)
    
    rf_train_ap = average_precision_score(train_labels, rf_train)
    rf_dev_ap = average_precision_score(dev_labels, rf_dev)
    
    rf_train_prec = precision_score(train_labels, rf_train > 0.5)
    rf_dev_prec = precision_score(dev_labels, rf_dev > 0.5)
    
    rf_train_recall = recall_score(train_labels, rf_train > 0.5)
    rf_dev_recall = recall_score(dev_labels, rf_dev > 0.5)

    print(random_search.best_estimator_)

        
    print(f'Train AUC:        {rf_train_auc:.4f}\n'
          f'Train AP:         {rf_train_ap:.4f}\n'
          f'Train Precision:  {rf_train_prec:.4f}\n'
          f'Train Recall:     {rf_train_recall:.4f}\n'
          f'Dev   AUC:        {rf_dev_auc:.4f}\n'
          f'Dev   AP:         {rf_dev_ap:.4f}\n'
          f'Dev   Precision:  {rf_dev_prec:.4f}\n'
          f'Dev   Recall:     {rf_dev_recall:.4f}')