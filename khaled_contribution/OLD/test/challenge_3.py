#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:25:02 2018

Name: khalednakhleh
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from timeit import default_timer as timer

def lasso_prediction(x, y):
    
    lasso = Lasso(normalize = "True", tol = 0.001, max_iter = 5000000)
    w, q = x.shape
    
    coefs = []
    preds = []
    
    i = 0
    
    while i < q:
        
        start = timer() 
        print("\n------------------------------------\n")
        name ="Fitting for country no.: %s" %(i + 1)
        print(name)
        
        y_revert = y.iloc[:, i]
        
        if x.iloc[0, i] < 4000000:
        
            alpha = 1
            lasso.set_params(alpha = alpha)
            population_vector = x.iloc[:, i]
            lasso.fit(x, population_vector)
        
            while np.count_nonzero(lasso.coef_) > 5:
                
                alpha = alpha + 10
                lasso.set_params(alpha = alpha)
                population_vector = x.iloc[:, i]
                y_i = y.drop([i], axis = 1)
                lasso.fit(x, population_vector)
         
            prediction = lasso.predict(y)
            coefs.append(lasso.coef_)
            preds.append(prediction)
            end_timer = timer() - start
        
            print("Alpha value: " + str(alpha))
            time_statement = "Fitting time for country no. %s: " %(i + 1)
            print(time_statement + str(round(end_timer, 4)) + " seconds.")
            print("Number of countries used for fitting: " + str(np.count_nonzero(lasso.coef_)))
            
        else:
            
            alpha = 1000
            lasso.set_params(alpha = alpha)
            population_vector = x.iloc[:, i]
            y_i = y.drop([i], axis = 1)
            lasso.fit(x, population_vector)
            
            while np.count_nonzero(lasso.coef_) > 5:
                
                alpha = alpha + 1000
                lasso.set_params(alpha = alpha)
                population_vector = x.iloc[:, i]
                y_i = y.drop([i], axis = 1)
                lasso.fit(x, population_vector)
                
            prediction = lasso.predict(y)
            coefs.append(lasso.coef_)
            preds.append(prediction)
            
            end_timer = timer() - start
        
            print("Alpha value: " + str(alpha))
            time_statement = "Fitting time for country no. %s: " %(i + 1)
            print(time_statement + str(round(end_timer, 4)) + " seconds.")
            print("Number of countries used for fitting: " + str(np.count_nonzero(lasso.coef_)))
            
        i = i + 1
        y.iloc[:, i] = y_revert
     
    return coefs, preds
    

def main():
    
    population_training_df = pd.read_csv('population_training.csv', encoding='cp1252')
    population_testing_df = pd.read_csv('population_testing.csv', encoding='cp1252')
    kaggle_file = pd.read_csv('population_sample_kaggle.csv', encoding='cp1252')
    parameter_file = pd.read_csv('population_parameters.csv', encoding='cp1252')
    
    population_training_df.drop(['Country Name'], axis=1, inplace=True)
    population_testing_df.drop(['Country Name'], axis=1, inplace=True)
    
    
    population_training_df = population_training_df.T
    population_training_df = population_training_df.T
    
    X = population_training_df.T
    Y = population_testing_df.T


    start = timer() 
    
    coefs, preds = lasso_prediction(X, Y)
    
    preds = np.transpose(preds)
    coefs = np.transpose(coefs)
    
    print("\n\n")
    print(X.shape)
    print(Y.shape)
    print(np.shape(preds))
    print(np.shape(coefs))
    
    preds_1 = pd.DataFrame(data = preds)
    coefs_1 = pd.DataFrame(data = coefs)
 
    preds_1.reindex(index = kaggle_file.index, columns = kaggle_file.columns)
    coefs_1.reindex(index = parameter_file.index, columns = parameter_file.columns)
    preds_1.to_csv("reformatted_preds.csv")
    coefs_1.to_csv("reformatted_coefs.csv")
 
    
    timer_end = timer() - start
    print("\n\nTotal elapsed time: " + str(timer_end) + " seconds.")



if __name__ == "__main__":
    main()
  
    
    
    
    
    
    
    
    
    
    
