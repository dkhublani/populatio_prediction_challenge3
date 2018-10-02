#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:23:54 2018

Name: khalednakhleh
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from timeit import default_timer as timer
from sklearn.metrics import mean_squared_error as mse


def lasso_pred(x, y):
    
    lasso = Lasso(normalize = "True", tol = 0.1, max_iter = 5000)
    w, q = x.shape
    
        
    coefs = []
    preds = []
    
    i = 0
    
    alphas = 10 ** (np.linspace(0.01, 8, 1000))
    
        
    population_vector = x.iloc[:, i]
    x.iloc[:, i] = np.zeros(w)
    lasso.set_params(alpha = 1000000000)
    y_true = y.iloc[:, i]
    lasso.fit(x, population_vector)    
    prediction = lasso.predict(y)
    error = mse(y_true, prediction)  
    x.iloc[:, i] = population_vector    
    
    for a in alphas:
            
        lasso.set_params(alpha = a)
        population_vector = x.iloc[:, i]
        x.iloc[:, i] = np.zeros(w)
        y_true = y.iloc[:, i]
        lasso.fit(x, population_vector)
        prediction = lasso.predict(y)
        mean_error = mse(y_true, prediction)   
      
        if (mean_error < error) and 0 < np.count_nonzero(lasso.coef_) <= 5:     
            coefs_best = lasso.coef_
            pred_best = prediction
            error = mean_error
            alpha_value = a
            country_number = np.count_nonzero(lasso.coef_)
     
    coefs.append(coefs_best)
    preds.append(pred_best)     
    print("Aplha value: " + str(alpha_value))
    print("Number of countries used for fitting: " + str(country_number))
    i = i + 1    
  
    return coefs, preds    
        
        
def main():
    
    population_training_df = pd.read_csv('population_training.csv', encoding='cp1252')
    population_testing_df = pd.read_csv('population_testing.csv', encoding='cp1252')
    kaggle_file = pd.read_csv('population_sample_kaggle.csv', encoding='cp1252')
    #parameter_file = pd.read_csv('population_parameters.csv', encoding='cp1252')
    
    population_training_df.drop(['Country Name'], axis=1, inplace=True)
    population_testing_df.drop(['Country Name'], axis=1, inplace=True)
    
    
    X = population_training_df.T
    Y = population_testing_df.T


    start = timer() 
    
    coefs, preds = lasso_pred(X, Y)
    
    #preds = np.transpose(preds)
    #coefs = np.transpose(coefs)
    
    print("\n\n")
    print(X.shape)
    print(Y.shape)
    
    """
    #print(np.shape(preds))
    #print(np.shape(coefs))
    
    preds_1 = pd.DataFrame(data = preds)
    coefs_1 = pd.DataFrame(data = coefs)
 
    preds_1.reindex(index = kaggle_file.index, columns = kaggle_file.columns)
    coefs_1.reindex(index = parameter_file.index, columns = parameter_file.columns)
    preds_1.to_csv("reformatted_preds.csv")
    coefs_1.to_csv("reformatted_coefs.csv")
 
    
    timer_end = timer() - start
    print("\n\nTotal elapsed time: " + str(timer_end) + " seconds.")


   """
if __name__ == "__main__":
    main()
  
    

