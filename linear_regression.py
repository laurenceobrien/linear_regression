### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def simulate_data():
    x0 = [1]*1000
    x1 = np.random.exponential(9000,1000)
    x2 = np.random.poisson(15, 1000)
    beta0 = 0
    beta1 = 10
    beta2 = -3
    epsilon = np.random.normal(0,1,1000)

    y = [None]*1000
    for i in range(0,1000):
        y[i] = beta1*x1[i] + beta2*x2[i] + epsilon[i]

    
    ind_vars = pd.DataFrame( data = x0, columns=["x0"])
    ind_vars['x1'] = x1
    ind_vars['x2'] = x2
   
    
    #print(y)
    #print(y[1])
    #print(x1[1], x2[1], epsilon[1])
    #print(vars)

    data = {"X": ind_vars, "beta" : [beta0, beta1, beta2], "y": y}
    #print(data)
    return data

    """
    Simulates data for testing linear_regression models.

    RETURNS
        data (dict) contains X, y, and beta vectors.
    """
    

data = simulate_data()
print(data)
print(data["X"])
def compare_models(ind_vars, y):

    stats_model = sm.OLS(y, ind_vars)
    stats_results = stats_model.fit()
    print(stats_results.params)

    sk_model = LinearRegression().fit(ind_vars, y)
    print(sk_model.coef_)

    results = pd.DataFrame(data = stats_results.params, columns=["statsmodels"])
    results.insert(1, "sklearn", sk_model.coef_)
    print(results)
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """
    
compare_models(data["X"], data["y"])


def load_hospital_data():



    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    pass


def prepare_data():
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    pass


def run_hospital_regression():
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass
 

### END ###