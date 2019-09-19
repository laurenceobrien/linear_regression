### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def simulate_data():
    x1 = np.random.exponential(9000,1000)
    x2 = np.random.poisson(15, 1000)
    beta1 = 10
    beta2 = -3
    epsilon = np.random.normal(0,1,1000)

    y = [None]*1000
    for i in range(0,1000):
        y[i] = beta1*x1[i] + beta2*x2[i] + epsilon[i]

    vars = pd.DataFrame( data = x1, columns=["x1"])
    vars.insert(1, "x2", x2)
    vars.insert(1, "epsilon", epsilon) 
    
    print(y)
    print(y[1])
    print(x1[1], x2[1], epsilon[1])
    print(vars)

    stats_model = 



    """
    Simulates data for testing linear_regression models.
    INPUT
        nobs (int) the number of observations in the dataset
    RETURNS
        data (dict) contains X, y, and beta vectors.
    """
    pass

simulate_data()
def compare_models():
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """
    pass


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