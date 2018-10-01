#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:23:54 2018

Name: khalednakhleh
"""

# Importing packages to be used.
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from timeit import default_timer as timer
from sklearn.metrics import mean_squared_error as mse

# Defining a function that gives values with required challenge 3 conditions.
def lasso_pred(x, y):
    
    # Initializing lasso regression model
    lasso = Lasso(normalize = "True", tol = 0.001, max_iter = 100000)
    w, q = x.shape
    
    # Creating empty arrays for coefficients and predictions    
    coefs = []
    preds = []
    
    # Country number
    i = 0
    
    # Beginning calculation for each country in the training set
    while i < q:   
        
        start = timer()
        print("\n------------------------------------\n")
        name ="Fitting for country no.: %s" %(i + 1)
        print(name)
        
        # Initializing values for comparsion with alpha vector
        population_vector = x.iloc[:, i]
        x.iloc[:, i] = np.zeros(w)
        lasso.set_params(alpha = 10 ** 10)
        y_true = y.iloc[:, i]
        lasso.fit(x, population_vector)    
        prediction = lasso.predict(y)
        error = mse(y_true, prediction)  
        x.iloc[:, i] = population_vector    
        
        # If country's population is low, then the first condition is used for 
        # small alpha value. If the population is larger than the given value, 
        # then the second condition will be used, will larger alpha values.
        
        if x.iloc[0, i] < 2500000:
            
            alphas = 10 ** (np.linspace(1, 4, 3000))
            
            for a in alphas:
                
                # Calculating prediction values  
                lasso.set_params(alpha = a)
                population_vector = x.iloc[:, i]
                x.iloc[:, i] = np.zeros(w)
                y_true = y.iloc[:, i]
                lasso.fit(x, population_vector)
                prediction = lasso.predict(y)
                mean_error = mse(y_true, prediction)  
                x.iloc[:, i] = population_vector
                
                # Saving the preferred alpha value that gives five countries
                if (mean_error < error) and np.count_nonzero(lasso.coef_) == 5:     
                    coefs_best = lasso.coef_
                    pred_best = prediction
                    error = mean_error
                    alpha_value = a
                    country_number = np.count_nonzero(lasso.coef_)
                    
            # Appending the correct coefficient and prediction values 
            coefs.append(coefs_best)
            preds.append(pred_best)    
            
            end_timer = timer() - start
            
            # Printing some info
            time_statement = "Fitting time for country no. %s: " %(i + 1)
            print(time_statement + str(round(end_timer, 3)) + " seconds.")
            print("Alpha value: " + str(alpha_value))
            print("Number of countries used for fitting: " + str(country_number))
            
            i = i + 1
            
        else:
            
            alphas = 10 ** (np.linspace(3, 7, 3000))
            
            for a in alphas:
                
                # Calculating prediction values   
                lasso.set_params(alpha = a)
                population_vector = x.iloc[:, i]
                x.iloc[:, i] = np.zeros(w)
                y_true = y.iloc[:, i]
                lasso.fit(x, population_vector)
                prediction = lasso.predict(y)
                mean_error = mse(y_true, prediction)  
                x.iloc[:, i] = population_vector
                
                # Saving the preferred alpha value that gives five countries
                if (mean_error < error) and np.count_nonzero(lasso.coef_) == 5:     
                    coefs_best = lasso.coef_
                    pred_best = prediction
                    error = mean_error
                    alpha_value = a
                    country_number = np.count_nonzero(lasso.coef_)
            
            # Appending the correct coefficient and prediction values 
            coefs.append(coefs_best)
            preds.append(pred_best)     
            
            end_timer = timer() - start
        
            # Printing some info
            time_statement = "Fitting time for country no. %s: " %(i + 1)
            print(time_statement + str(round(end_timer, 3)) + " seconds.")
            print("Alpha value: " + str(alpha_value))
            print("Number of countries used for fitting: " + str(country_number))
            
            i = i + 1
  
    return coefs, preds    
        
# Initializing the program        
def main():
    
    # importing and preparing the .csv data files 
    population_training_df = pd.read_csv('population_training.csv', encoding='cp1252')
    population_testing_df = pd.read_csv('population_testing.csv', encoding='cp1252')
    kaggle_file = pd.read_csv('population_sample_kaggle.csv', encoding='cp1252')
    parameter_file = pd.read_csv('population_parameters.csv', encoding='cp1252')
    
    # Dropping country name column to have just the data
    population_training_df.drop(['Country Name'], axis=1, inplace=True)
    population_testing_df.drop(['Country Name'], axis=1, inplace=True)
    
    # Transposing training data and testing data to have a 40 x 258 and 17 x 258 shape
    X = population_training_df.T
    Y = population_testing_df.T

    # starting timer
    start = timer() 
    
    # Calculating the coefficients and predictions based on lasso_pred function
    coefs, preds = lasso_pred(X, Y)
    
    # Transposing to fit the challenge's criteria
    preds = np.transpose(preds)
    coefs = np.transpose(coefs)
    
    # Printing shape of training and testing data, as well as coefficients and predictions
    print("\n\n")
    print(X.shape)
    print(Y.shape)
    print(np.shape(coefs))
    print(np.shape(preds))

    # Saving to pandas dataframes
    preds_1 = pd.DataFrame(data = preds)
    coefs_1 = pd.DataFrame(data = coefs)
    
    # Reindexing the file to fit the challenge's format
    preds_1.reindex(index = kaggle_file.index, columns = kaggle_file.columns)
    coefs_1.reindex(index = parameter_file.index, columns = parameter_file.columns)
    preds_1.to_csv("predictions.csv")
    coefs_1.to_csv("coefficients.csv")
 
    # Calculating and printing elapsed time
    timer_end = timer() - start
    print("\n\nTotal elapsed time: " + str(timer_end) + " seconds.")


if __name__ == "__main__":
    main()
  
