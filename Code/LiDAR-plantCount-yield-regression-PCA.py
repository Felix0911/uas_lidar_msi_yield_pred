# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Fei Zhang

Code description:
    1. read the data;
    2. stepwise regression;
    3. lasso regression;
    4. random forest regression; machine learning algorithm, dataset was 
    separate into train and test.

Version: 1.0

Reference:
"""



'''Set working directory and all files' paths'''
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


import time
start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from scipy import stats
import statsmodels.tools as st
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.decomposition import PCA


"""===========================MAIN PROGRAM BELOW============================"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor    

#%%
#predicted R-squared
#ref1: https://gist.github.com/benjaminmgross/d71f161d48378d34b6970fa6d7378837

def pressStatistic(y_true, y_pred, xs):
    """
    Calculation of the `Press Statistics https://en.wikipedia.org/wiki/PRESS_statistic`_
    ref2: https://statisticaloddsandends.wordpress.com/2018/07/30/the-press-statistic-for-linear-regression/
    ref3: https://www.youtube.com/watch?v=SBYO0dPbAFo&ab_channel=statisticsmatt
    
    Inputs:
    1. the y_true is the ground truth response;
    2. the y_pred is the predicted response from all observations by `the` 
    regression model.
    3. the xs is the array of sample_size*feature_dim obervations/subjects
    
    Output: the PRESS statistic, a scalar.
    """
    res = y_pred - y_true
    hat = xs.dot(np.linalg.pinv(xs))
    den = (1 - np.diagonal(hat))
    sqr = np.square(res/den)
    return sqr.sum()

def predictedR2(y_true, y_pred, xs):
    """
    Calculation of the `Predicted R-squared <https://rpubs.com/RatherBit/102428>`_
    """
    press = pressStatistic(y_true=y_true,
                            y_pred=y_pred,
                            xs=xs
    )
    print(f"PRESS statistic = {press:.3f}")
    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - press / sst
 
def ordinaryR2(y_true, y_pred):
    """
    Calculation of the unadjusted r-squared, goodness of fit metric
    """
    sse  = np.square( y_pred - y_true ).sum()
    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - sse/sst



def printModel(intercept, coeffs, coeff_names, pred_name):
    model = f'{pred_name} = {intercept}'
    for coeff, name in zip(coeffs, coeff_names):
        model += f'+ {coeff}*{name}'
    print(model)

def linearFit(X_select, y, pred_name):
    reg = LinearRegression(fit_intercept=True).fit(X_select, y)
    coeff = np.round(reg.coef_, 3)
    intercept = np.round(reg.intercept_, 3)
    printModel(intercept, coeff, X_select.columns, pred_name)    
    
    #calculate RMSE, R2, adjusted R2
    y_pred = reg.predict(X_select)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    nrmse = rmse/np.mean(y)
    r2 = reg.score(X_select, y)
    n = len(y)
    # adjusted_r2 = 1- (1-r2)*(n-1)/(n-k_best-1)
    adjusted_r2 = 1- (1-r2)*(n-1)/(n-X_select.shape[1]-1)
    predicted_r2 = predictedR2(y_true=y, y_pred=y_pred, xs=X_select)
    print(f'R2:{r2:.3f}\n'
          f'adjusted R2:{adjusted_r2:.3f}\n'
          f'predicted R2:{predicted_r2:.3f}\n'
          f'RMSE:{rmse:.3f}\n'
          f'normalized RMSE: {nrmse:.3f}\n'
          f"y_mean: {np.mean(y):.3f}")
    
    return y_pred

def normX(train_data):
    '''compute the training mean across feature dimensions and the training
    standard deviation, and then normalize by subtracting the training mean 
    from both the train and test sets, and then divide both sets by the train 
    standard deviation.'''
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)
    train_data_update = (train_data - train_mean) / train_std
    
    return train_data_update


def plotNormResidual(y_pred, residual, saveFlag=False, title_str = None):
    '''Plot residuals versus fit and normal probability plot of residuals.'''
    px = y_pred
    py = residual
    plt.figure()
    plt.scatter(px, py)
    plt.hlines(y=0, xmin = min(px), xmax=max(px), colors = 'r', linestyles ='dashed')
    plt.xlabel("Fitted value")
    plt.ylabel("Residual")
    plt.title(f"{title_str}: Residual versus fits")
    plt.grid()    
    if saveFlag:
        plt.savefig(f"Residual versus fits {title_str}.jpg")
    
    
    plt.figure()
    npx = residual
    stats.probplot(npx, plot=plt)
    plt.grid()
    plt.xlabel("Normal quantile")
    plt.ylabel("Residual")
    plt.title(f"{title_str}: Normal probability plot")
    if saveFlag:
        plt.savefig("Normal probability plot {title_str}.jpg")
        
    return True


def readLiDARData(file_path, ifPrint=True, pred='yield'):
    '''
    Read the samples and the ground truth from the excel file.
    Output: X - n_sample * n_variable, pandas dataframe; y - n_sample, pandas series
    '''
    org_df = pd.read_excel(file_path, sheet_name=0)
    col_names_temp = org_df.columns[:7]
    
    if pred=='yield':
        dep_var = org_df.columns[-1]
    elif pred=='PlantCount':
        dep_var = org_df.columns[-2]
    else:
        raise ValueError('Please assign the correct name of the dependent variable.')

    if ifPrint:
        print(f'Columns names = {col_names_temp}')
        print(f'Dependent variable = {dep_var}')
    X = org_df.iloc[:, :7].astype('float')
    y = org_df[dep_var].astype('float')
    X.index = np.arange(X.shape[0])
    y.index = np.arange(y.shape[0])
    
    return X, y, col_names_temp


def readMSIData(file_path, ifPrint=True, pred='yield'):
    '''
    Read the samples and the ground truth from the excel file.
    Output: X - n_sample * n_variable, pandas dataframe; y - n_sample, pandas series
    '''
    org_df = pd.read_excel(file_path, sheet_name=0)
    col_names_temp = org_df.columns[7:-2]
    
    def shortenName(name_str):
        '''
        Take the initials of the words and return the initials.
        '''
        name_ls = name_str.split(' ')
        if len(name_ls)==1:
            return name_str
        else:
            initials = "".join([name[0] for name in name_ls])
            return initials
    
    col_names_initial = [shortenName(name_t) for name_t in col_names_temp]
        
    if pred=='yield':
        dep_var = org_df.columns[-1]
    elif pred=='PlantCount':
        dep_var = org_df.columns[-2]
    else:
        raise ValueError('Please assign the correct name of the dependent variable.')

    if ifPrint:
        print(f'Columns names = {col_names_temp}')
        print(f'Columns initial = {col_names_initial}')
        print(f'Dependent variable = {dep_var}')
        
    X = org_df.iloc[:, 7:-2].astype('float')
    y = org_df[dep_var].astype('float')
    X.columns = col_names_initial 
    y.name = pred
    X.index = np.arange(X.shape[0])
    y.index = np.arange(y.shape[0])
    
    return X, y, col_names_temp

def readLiDARMSIData(file_path, ifPrint=True):
    '''
    Read the samples and the ground truth from the excel file.
    Output: X - n_sample * n_variable, pandas dataframe; y - n_sample, pandas series
    '''
    org_df = pd.read_excel(file_path, sheet_name=0)
    col_names_temp = org_df.columns
    
    def shortenName(name_str):
        '''
        Take the initials of the words and return the initials.
        '''
        name_ls = name_str.split(' ')
        if len(name_ls)==1:
            return name_str
        else:
            initials = "".join([name[0] for name in name_ls])
            return initials
    
    col_names_initial = [shortenName(name_t) for name_t in col_names_temp]
    X_y = org_df
    X_y.columns = col_names_initial    
    if ifPrint:
        print(f'Columns names = {col_names_temp}')
        print(f'Columns initial = {col_names_initial}')
        print(f'Dependent variable = {col_names_temp[-1]}')
    X = X_y.iloc[:, :-2].astype('float')
    y = X_y['ykh'].astype('float')
    X.index = np.arange(X.shape[0])
    y.index = np.arange(y.shape[0])
    
    return X, y, col_names_initial, col_names_temp[-1]


def forward_stepwise(X, y, threshold_in, verbose=False):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded, dtype='float64')
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included

def backward_stepwise(X, y, threshold_out, verbose=False):
    included=list(X.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included # a list of included feature names.\

def myStepwise(X, y, step_alpha, verbose=False):
    #stepwise regression
    print('\n stepwise regression')
    #if n<p, use forward selection, otherwise, use backward selection
    #ref: http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/154-stepwise-regression-essentials-in-r/
    if X.shape[0] < X.shape[1]: 
        included = forward_stepwise(X, y, step_alpha, verbose=verbose)
    else:
        included = backward_stepwise(X, y, step_alpha, verbose=verbose)
    #re-fit the model based on the stepwise regression result
    fit_X = sm.add_constant(pd.DataFrame(X[included]))
    stepwise_model = sm.OLS(y, fit_X).fit()
    predictions = stepwise_model.predict(fit_X)
    rmse_step = np.round(st.eval_measures.rmse(y, predictions), decimals=3)
    nrmse = rmse_step/np.mean(y)
    predicted_r2 = predictedR2(y_true=y, y_pred=predictions, xs=X[included])
    print(stepwise_model.summary())
    print(f'RMSE={rmse_step:.3f}.\n'
          f'normalized RMSE: {nrmse:.3f}\n'
          f'predicted R2={predicted_r2:.3f}\n')
    
    return predictions, X[included]

def fitPCA(X, var_th=0.9, verbose=False):
    '''
    Input: X -  pandas dataframe of (n_samples, n_features)
    Output: X_new_df - pandas dataframe of (n_samples, selected_n_components)
    Function: Fit PCA on X and keep selected_n components so that they explain
    more than 90 percent variance. Then transform the X to X_new and make it 
    a dataframe at last.
    
    '''
    n_pc = min(X.shape)
    pca = PCA(n_components=n_pc)
    pca.fit(X)
    var_ratio = pca.explained_variance_ratio_
    for i in range(len(var_ratio)):
        sum_ratio = sum(var_ratio[:i])
        if sum_ratio>var_th:
            if verbose:
                print(f'Number of selected components: {i+1}')
            pca_refit = PCA(n_components=i+1)
            pca_refit.fit(X)
            X_new = pca_refit.fit_transform(X)
            X_new_df = pd.DataFrame(X_new, columns=[f'PC{j+1}' for j in range(i+1)])
            break
    
    return X_new_df
#%%
#single file
flight_stamp = ['07281204', '07311057', '08061055', 
                '08101049', '08141148', '08211044', '08241132']
fl_stp = flight_stamp[6]
file_path = fr'E:\2020snapbeans\2020{fl_stp[:4]}\{fl_stp}_LiDAR_MSI_combine-v2.xlsx'

out_name = 'yield' #set as 'PlantCount' or 'yield'
# X, y, X_names = readLiDARData(file_path, ifPrint=True, pred=out_name) #make sure the data type is float for X and y!!!!!!
# X, y, X_names = readMSIData(file_path, ifPrint=True, pred=out_name) #make sure the data type is float for X and y!!!!!!
X, y, X_names, _ = readLiDARMSIData(file_path, ifPrint=True) 

X_new_df = fitPCA(X, verbose=True)
#%%
y_pred, X_select = myStepwise(X_new_df, y, step_alpha=0.05)

linearFit(X_select, y, out_name)



#%%
print("--- %.1f seconds ---" % (time.time() - start_time))