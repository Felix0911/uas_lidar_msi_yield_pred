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


def removeCollinearVar(X, thresh=5.0):
    #remove colinear variables based on the variance inflation factor (VIF)
    #ref: https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-pandas-columns-in-python
    variables = list(range(X.shape[1]))
    print(variables)
    dropped = True
    while dropped and len(variables)>1:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
                for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True


    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]



def selectKBestFeature_F_Rgr(X, y, k_best=4):
    print("\n\n Results from SelectKBestFeature_F_Rgr:")
    k_best = 4
    bestfeatures = SelectKBest(f_regression, k=k_best)
    fit = bestfeatures.fit(X.to_numpy(),y.to_numpy())
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    best_features = featureScores.nlargest(k_best,'Score')
    print(best_features)  #print 10 best features
    X_select = bestfeatures.transform(X)
    coeff_names = list(best_features.iloc[:,0])
    X_select_df = pd.DataFrame(X_select, columns=coeff_names)
    
    return X_select_df, coeff_names



def recursiveFeatureElimination_SVR(X, y, k_best=4, step=2):
    print("\n\n Results from recursiveFeatureElimination_SVR:")
    estimator = SVR(kernel='linear') #other kernels does not expose "coef_" or "feature_importances_" attributes
    selector = RFE(estimator, n_features_to_select=k_best, step=step)
    selector = selector.fit(X, y)
    best_ft_mask = selector.support_
    X_select = X.loc[:,best_ft_mask]
    coeff_names = X.columns[best_ft_mask]
    # rfe_score = selector.score(X,y) #r2_score from SVR regression 
    # print(f'RFE score: {rfe_score:.3f}')
    
    return X_select, coeff_names

def printModel(intercept, coeffs, coeff_names):
    model = f'yield = {intercept}'
    for coeff, name in zip(coeffs, coeff_names):
        if coeff<0:
            model += f'{coeff}*{name}'
        else:
            model += f'+{coeff}*{name}'
    print(model)

def linearFit(X_select, y):
    reg = LinearRegression(fit_intercept=True).fit(X_select, y)
    coeff = np.round(reg.coef_, 3)
    intercept = np.round(reg.intercept_, 3)
    printModel(intercept, coeff, X_select.columns)    
    
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

def cleanDataFrame(X):
    var_per_feat = np.var(X, axis=0)
    feat_mask = np.asarray(var_per_feat>0.001)
    out_X = X.iloc[:, feat_mask]
    return out_X

def normX(X):
    '''compute the training mean across feature dimensions and the training
    standard deviation, and then normalize by subtracting the training mean 
    from both the train and test sets, and then divide both sets by the train 
    standard deviation.'''
    # X_clean = cleanDataFrame(X)
    train_mean = np.mean(X, axis=0)
    train_std = np.std(X, axis=0)
    train_data_update = (X - train_mean) / train_std
    
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


def mylassoCV2(X, y, out_name, input_names, selectFtNum = 5, 
               max_iter = 10000,  selectFtFlag=False, 
               normalize=False, cv= 5, plotFlag = True):
    print("Before feature selection: ")
    X_np = np.array(X)
    y_np = np.array(y)
    clf = linear_model.LassoCV(max_iter = max_iter, cv= cv)
    clf.fit(X_np, y_np)
    
    #print the full model as an equation
    print('The full model is:')
    dv_indx = np.nonzero(clf.coef_)[0] #indices of the non-zero coefficient, 1-d numpy array
    
    if len(dv_indx)==0: #if no feature were selected, 
        print("LassoCV fail to converge!")
        return 0, 0, 0
    
    
    model_content = out_name + " = " + str(round(clf.intercept_, 3))
    single_out_pred_name = []
    for indx in dv_indx:
        single_out_pred_name.append(input_names[indx])
        model_content = model_content + " + " + str(round(clf.coef_[indx], 3)) + " " + input_names[indx]
    print(model_content)
    
    
    y_pred = clf.predict(X_np)
    rmse = np.sqrt(metrics.mean_squared_error(y_np, y_pred))
    nrmse = rmse/np.mean(y_np)
    residual = y_pred - y_np
    r_sq = clf.score(X_np, y_np)
    r_sq_adj = 1-(1-r_sq)*(len(y_np)-1)/(len(y_np)-len(dv_indx)-1) #1-(1-R2)*(n-1)/(n-k-1)
    predicted_r2 = predictedR2(y_true=y_np, y_pred=y_pred, xs=X_np[:,dv_indx])
    print(f"R-sq={r_sq:.3f}; \n"
          f"R-sq-adj={r_sq_adj:.3f}; \n"
          f'predicted R2={predicted_r2:.3f}\n'
          f"RMSE={rmse:.3f}. normalized RMSE: {nrmse:.3f}\n"
          f"y_mean: {np.mean(y_np):.3f}\n")  
        
    if selectFtFlag:
        #####
        #select specific number of features
        print('\n After feature selection: ')
        importance = np.abs(clf.coef_)
        idx_features = (-importance).argsort()[:selectFtNum] #keep only a number of features
        select_ft_coef = clf.coef_[idx_features]
        select_ft_coef = np.array([num for num in select_ft_coef if num!=0])
        if len(select_ft_coef) < selectFtNum: #if the model give less features than selected, reset the number of select features
            idx_features = idx_features[:len(select_ft_coef)]
        
        name_features = np.array(input_names)[idx_features]
        print('Selected features: {}'.format(name_features))
        
        # #calculate the y_pred from the truncated LASSO regression model
        # y_pred = clf.intercept_ 
        # for select_coef, idx in zip(select_ft_coef, idx_features):
        #     y_pred += select_coef*X_np[:,idx]
            
        #calculate the y_pred from the linear regression model. 
        #(better than directly use truancated model)
        X_select = X.iloc[:, idx_features]
        y_pred = linearFit(X_select, y_np)
        
        #remove the collinear variables in the truncated model and then redo
        #the linear regression.
        print('\nRemove collinear variables from the truncated model:')
        X_select_clean = removeCollinearVar(X_select, thresh=5.0)
        linearFit(X_select_clean, y_np)
    
        
    
    if plotFlag:
        plotNormResidual(y_pred, residual, saveFlag=False, title_str = 'LassoCV')
    return r_sq, rmse, model_content

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
    if feat_base == 'LiDAR':
        X = org_df.iloc[:, -9:-1].astype('float')
        X.columns = col_names_initial[-9:-1]   
    elif feat_base == 'MSI':
        X = org_df.iloc[:, 2:-10].astype('float')
        X.columns = col_names_initial[2:-10]
    elif feat_base == 'MSI-LiDAR':
        X = org_df.iloc[:, 2:-1].astype('float')
        X.columns = col_names_initial[2:-1]    
    else:
        raise ValueError('Wrong feature base name!!!')
    
    
    if ifPrint:
        print(f'Columns names = {col_names_temp}')
        print(f'Columns initial = {col_names_initial}')
        print(f'Dependent variable = {col_names_temp[-1]}')
    
    y = org_df['yield'].astype('float')
    X = cleanDataFrame(X)
    #remove the bad cultivar plot in 2019 data
    plot_mask = np.ones(X.shape[0]).astype(bool)
    plot_mask[4] = False 
    y = y[plot_mask]
    X = X.iloc[plot_mask, :]
    X.index = np.arange(X.shape[0])
    y.index = np.arange(y.shape[0])
    
    return X, y, X.columns

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



def randomForest(X, y, test_size=0.4, plotFlag=True):
    '''ref: https://mljar.com/blog/random-forest-overfitting/'''
    rmse_train_ls = []
    rmse_test_ls = []
    r_sq_train_ls = []
    r_sq_test_ls = []
    for iter in range(100):
        '''take average over multiple runs to eliminate the impact of initial randoms'''
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size)
        rf_regr = RandomForestRegressor(n_estimators=100)
        rf_regr.fit(X_train, y_train)
        y_train_pred = rf_regr.predict(X_train)
        y_test_pred = rf_regr.predict(X_test)
        rmse_train = np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred))
        residual = y_test_pred - y_test
        r_sq_train = rf_regr.score(X_train, y_train)
        r_sq_test = rf_regr.score(X_test, y_test)
        rmse_train_ls.append(rmse_train)
        rmse_test_ls.append(rmse_test)
        r_sq_train_ls.append(r_sq_train)
        r_sq_test_ls.append(r_sq_test)
    
    ave_rmse_train = np.mean(np.array(rmse_train_ls))
    ave_rmse_test = np.mean(np.array(rmse_test_ls))
    ave_Rsq_train = np.mean(np.array(r_sq_train_ls))
    ave_Rsq_test = np.mean(np.array(r_sq_test_ls))
    
    if plotFlag:
        plotNormResidual(y_test_pred, residual, saveFlag=False, title_str = None)
    
    return ave_rmse_train, ave_rmse_test, ave_Rsq_train, ave_Rsq_test


def trainRegressionModels(file_path, 
                          feat_base='LiDAR',
                          step_alpha = 0.01, 
                          step_VIF_th = 5.00,
                          max_iter = 10000, 
                          selectFtNum = 5, 
                          selectFtFlag = False,
                          normalize=False, 
                          cv= 5, 
                          trainLasso=True, 
                          trainSW=True, 
                          trainRF=True, 
                          trainFRgr=True,
                          trainRFE_SVR=True,
                          plotFlag = True):
    
    
    X, y, X_names = readLiDARMSIData(file_path, ifPrint=True) #make sure the data type is float for X and y!!!!!!
    if normalize:
        X = normX(X) #do not use the normalize parameter in LassoCV, otherwise it will cause a mess for feature selection
    
    out_name = 'yield'
    if trainLasso:
    #implementing LASSO; https://towardsdatascience.com/stopping-stepwise-why-
    #stepwise-selection-is-bad-and-what-you-should-use-instead-90818b3f52df.
        
        print('\n lasso cross validation')
        r_sq, rmse, model_content = mylassoCV2(X, y, out_name, X_names, 
                                               selectFtNum = selectFtNum, 
                                               max_iter = max_iter,  
                                               # normalize=normalize, 
                                               cv= cv, 
                                               selectFtFlag = selectFtFlag, 
                                               plotFlag = plotFlag)
        print("=================LASSO =========================")
        
    if trainSW:
        
        y_pred, X_select = myStepwise(X, y, step_alpha)
        
        print('\nRemove collinear variables from the truncated model:')
        X_select_clean = removeCollinearVar(X_select, thresh=step_VIF_th)
        linearFit(X_select_clean, y)
        print("==================Stepwise ================================")
        if plotFlag:
            residual = y_pred - y
            plotNormResidual(y_pred, residual, saveFlag=False, title_str = 'step_wise_forward')    

    if trainRF:
        #Random forest; it is not actually very suitable for small dataset.
        print("==================Random forest======================")
        print('\n random forest')
        X_np = np.array(X)
        y_np = np.array(y)
        rmse_train, rmse_test, r_sq_train, r_sq_test = randomForest(X_np, y_np, test_size=0.4)    
        print(f"R_sq_train={r_sq_train:.3f}; R_sq_test={r_sq_test:.3f};\n"
              f"RMSE_train={rmse_train:.3f}; RMSE_test={rmse_test:.3f}\n"
              f"y_mean: {np.mean(y):.3f}")   
    
    if trainFRgr:
        print("===================K best feature======================")
        X_select, coeff_names = selectKBestFeature_F_Rgr(X, y, selectFtNum)    
        y_pred = linearFit(X_select, y)
        if plotFlag:
            residual = y_pred - y
            plotNormResidual(y_pred, residual, saveFlag=False, title_str = 'SelectKbest_F_regression')    

        
    if trainRFE_SVR:
        
        X_select, coeff_names = recursiveFeatureElimination_SVR(X, y, selectFtNum, step=1)
        y_pred = linearFit(X_select, y)
        
        print('\nRemove collinear variables from the truncated model:')
        X_select_clean = removeCollinearVar(X_select, thresh=5.0)
        linearFit(X_select_clean, y)
        print("===================SVR======================")
        if plotFlag:
            residual = y_pred - y
            plotNormResidual(y_pred, residual, saveFlag=False, title_str = 'RFE_SVR')    

    return True
#%%
# For 2019
time_stamps = ['08051158', '08121245', '08141222', '08161204', '08201154']
tstp = time_stamps[3]
file_path = rf'G:\My Drive\papers\myWritings\2022YieldPrediction\Data\CombinedLiDARMSI\{tstp}_2019.xlsx'

# time_stamps = ['07281204', '07311057', '08061055', 
#                 '08101049', '08141148', '08211044', '08241132']
# tstp = time_stamps[0]
# file_path = rf'G:\My Drive\papers\myWritings\2022YieldPrediction\Data\CombinedLiDARMSI\{tstp}_2020.xlsx'




feat_base_ls = ['LiDAR', 'MSI', 'MSI-LiDAR']
feat_base = feat_base_ls[0]
trainRegressionModels(file_path, 
                      step_alpha = 0.05,
                      step_VIF_th = 5.00,
                      max_iter = 1000_000,
                      normalize=True,
                      selectFtFlag=True,
                      selectFtNum=5, #Use a large number to let the VIF decide how many variables should be included in the final model
                      trainLasso=True,
                      trainSW=False, 
                      trainRF=False,
                      trainFRgr=False,
                      trainRFE_SVR=False,
                      plotFlag = False)




#%%
print("--- %.1f seconds ---" % (time.time() - start_time))