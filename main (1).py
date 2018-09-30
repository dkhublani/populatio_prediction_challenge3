#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:26:40 2018

Name: khalednakhleh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse

def lasso_reg(alphas, x, y):
    
    lasso = Lasso(normalize = "True", tol = 0.001, max_iter = 50000)
    
    w, q = x.shape
    f = 1
    
    error = []
    
    for a in alphas:
        
        coefs = []
        preds = []
        
        lasso.set_params(alpha = a)
        i = 0
       
        while i < q:
            
            
            population_vector = x.iloc[:, i]
            train1=x.drop([i],axis=1)
            lasso.fit(train1, population_vector)
            test1=y.drop([i],axis=1)
            prediction = lasso.predict(test1)
            coefs.append(lasso.coef_)
            preds.append(prediction)
            i = i+1
            
        name_coef = "coef_no_%s.csv" %f
        name_preds = "pred_no_%s.csv" %f          
        
        np.savetxt(name_coef, coefs, delimiter=",")
        np.savetxt(name_preds, preds, delimiter=",")
        
        y_true = y.values
        
        result = mse(y_true, np.transpose(preds))
        error.append(result)
        
        f = f+1  
    
    return lasso, preds, error

def main():
    
    alphas = 10 ** np.linspace(1,6,5)
    
    population_training_df = pd.read_csv('population_training.csv', encoding='cp1252')
    population_testing_df = pd.read_csv('population_testing.csv', encoding='cp1252')
    
    population_training_df.drop(['Country Name'], axis=1, inplace=True)
    population_testing_df.drop(['Country Name'], axis=1, inplace=True)
    
    population_training_df = population_training_df.T
    population_training_df.fillna(population_training_df.median(),inplace = True)
    population_training_df = population_training_df.T
    
    X = population_training_df.T
    Y = population_testing_df.T

    print(X.shape)
    print(Y.shape)
    
   # returns prediction and error values of the last iteration
    model, prediction, error = lasso_reg(alphas, X, Y)
    
    # Graphing for Canada in pred_no_5 file and testing file
    prediction1=pd.read_csv('pred_no_2.csv', header=None)
    prediction1=prediction1.T
    plt.plot(prediction1.iloc[:,33], Y.iloc[:,33])
    plt.title("True vs. predicted of Canada")
    plt.xlabel("Predicted values")
    plt.ylabel("True values" )
    plt.savefig("canada_pred")
    plt.show()
    
    # Graphing the error rate vs. alpha value
    
    plt.plot(alphas, error)
    plt.title("error rate vs. alpha value")
    plt.xlabel("alpha value")
    plt.ylabel("mean square error")
    plt.savefig("alpha_vs_pred")
    plt.show()
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    