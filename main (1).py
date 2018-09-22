#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:59:32 2018

Name: khalednakhleh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

def fit(x, y):
    
    a = 50000000
    
    coefs = []
    model = Lasso(tol = 0.02, max_iter = 500000)
    x = x.T
    w, q = x.shape

    i = 0

    while i < q:
        t = x.iloc[:,i]
        x_i = x[x != t]
        model.set_params(alpha = a)
        model.fit(x_i, t)
        coefs.append(model.coef_)
        i = i + 1
    
    
    return coefs

  
def main():
    
    population_training_df = pd.read_csv('population_training.csv', encoding='cp1252')
    population_testing_df = pd.read_csv('population_testing.csv', encoding='cp1252')
    
    population_training_df.drop(['Country Name'], axis=1, inplace=True)
    population_testing_df.drop(['Country Name'], axis=1, inplace=True)
    
    population_training_df = population_training_df.T
    population_training_df.fillna(population_training_df.mean(),inplace = True)
    population_training_df = population_training_df.T
    
    X = population_training_df
    Y = population_testing_df
    
    print(X.shape)
    print(Y.shape)
    
    coefs = fit(X, Y)
    
    print(len(coefs))
    
    #prediction = predict(Y, coefs)
    
if __name__ == "__main__":
    
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    