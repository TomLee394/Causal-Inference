#Code by Kellie Ottoboni's package pscore_match http://www.kellieottoboni.com/pscore_match/_modules/pscore_match/match.html
"""
This module implements several variants of matching: 
one-to-one matching, one-to-many matching, with or without a caliper, 
and without or without replacement. 
Variants of the methods are examined in Austin (2014).


Austin, P. C. (2014), A comparison of 12 algorithms for matching on the 
propensity score. Statistic. Med., 33: 1057-1069.
"""

from __future__ import division
import numpy as np
import scipy
from scipy.stats import binom, hypergeom, gaussian_kde, ttest_ind, ranksums
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

################################################################################
################################# utils ########################################
################################################################################

def set_caliper(caliper_scale, caliper, propensity):
    # Check inputs
    if caliper_scale == None:
        caliper = 0
    if not(0<=caliper<1):
        if caliper_scale == "propensity" and caliper>1:
            raise ValueError('Caliper for "propensity" method must be between 0 and 1')
        elif caliper<0:
            raise ValueError('Caliper cannot be negative')

    # Transform the propensity scores and caliper when caliper_scale is "logit" or None
    if caliper_scale == "logit":
        propensity = np.log(propensity/(1-propensity))
        caliper = caliper*np.std(propensity)
    
    print('Caliper size: ', caliper)
    return caliper
    
def recode_groups(groups, propensity):
    # Code groups as 0 and 1
    groups = (groups == groups.unique()[0])
    N = len(groups)
    N1 = groups[groups == 1].index
    N2 = groups[groups == 0].index
    g1 = propensity[groups == 1]
    g2 = propensity[groups == 0]
    # Check if treatment groups got flipped - the smaller should correspond to N1/g1
    if len(N1) > len(N2):
       N1, N2, g1, g2 = N2, N1, g2, g1
    return groups, N1, N2, g1, g2

################################################################################
############################# Base Matching Class ##############################
################################################################################

class Match(object):
    """
    Parameters
    ----------
    groups : array-like 
        treatment assignments, must be 2 groups
    propensity : array-like 
        object containing propensity scores for each observation. 
        Propensity and groups should be in the same order (matching indices)    
    """
    
    def __init__(self, groups, propensity):
        self.groups = pd.Series(groups)
        self.propensity = pd.Series(propensity)
        assert self.groups.shape==self.propensity.shape, "Input dimensions dont match"
        assert all(self.propensity >=0) and all(self.propensity <=1), "Propensity scores must be between 0 and 1"
        assert len(np.unique(self.groups)==2), "Wrong number of groups"
        self.nobs = self.groups.shape[0]
        self.ntreat = np.sum(self.groups == 1)
        self.ncontrol = np.sum(self.groups == 0)
        
    def create(self, method='one-to-one', **kwargs):
        """
        Parameters
        ----------
        method : string
            'one-to-one' (default) or 'many-to-one'
        caliper_scale: string
            "propensity" (default) if caliper is a maximum difference in propensity scores,
            "logit" if caliper is a maximum SD of logit propensity, or "none" for no caliper
        caliper : float
             specifies maximum distance (difference in propensity scores or SD of logit propensity) 
        replace : bool
            should individuals from the larger group be allowed to match multiple individuals in the smaller group?
            (default is False)
    
        Returns
        -------
        A series containing the individuals in the control group matched to the treatment group.
        Note that with caliper matching, not every treated individual may have a match.
        """

        if method=='many-to-one':
            self._match_many(**kwargs)
            self._match_info()
        elif method=='one-to-one':
            self._match_one(**kwargs)
            self._match_info()
        else:
            raise ValueError('Invalid matching method')

    def _match_one(self, caliper_scale=None, caliper=0.05, replace=False):
        """
        Implements greedy one-to-one matching on propensity scores.

        Parameters
        ----------
        caliper_scale: string
            "propensity" (default) if caliper is a maximum difference in propensity scores,
            "logit" if caliper is a maximum SD of logit propensity, or "none" for no caliper
        caliper : float
             specifies maximum distance (difference in propensity scores or SD of logit propensity) 
        replace : bool
            should individuals from the larger group be allowed to match multiple individuals in the smaller group?
            (default is False)
        """
        caliper = set_caliper(caliper_scale, caliper, self.propensity)
        groups, N1, N2, g1, g2 = recode_groups(self.groups, self.propensity)
        
        # Randomly permute the smaller group to get order for matching
        morder = np.random.permutation(N1)
        matches = {}

        for m in morder:
            dist = abs(g1[m] - g2)
            if (dist.min() <= caliper) or not caliper:
                matches[m] = dist.idxmin()    # replace argmin() with idxmin()
                if not replace:
                    g2 = g2.drop(matches[m])
        self.matches = matches
        self.weights = np.zeros(self.nobs)
        self.freq = np.zeros(self.nobs)
        mk = list(matches.keys())
        mv = list(matches.values())
        for i in range(len(matches)):
            self.freq[mk[i]] += 1
            self.weights[mk[i]] += 1
            self.freq[mv[i]] += 1
            self.weights[mv[i]] += 1 

    

    def _match_info(self):
        """
        Helper function to create match info
        """
        assert self.matches is not None, 'No matches yet!'
        
#         self.matches = {
#             'match_pairs' : self.matches,
# #             'treated' : np.unique(list(self.matches.keys())),
# #             'control' : np.unique(list(self.matches.values()))
#         }
        self.treated = { 'treated' : list(np.unique(list(self.matches.keys())))
        
        }
        self.control = {
            'control' : np.unique(self.matches.values())
        }
#         self.matches['dropped'] = np.setdiff1d(list(range(self.nobs)), 
#                                     np.append(self.matches['treated'], self.matches['control']))

    


################################################################################
############################ helper funcs  #####################################
################################################################################

def whichMatched(matches, data, show_duplicates = True):
    """ 
    Simple function to convert output of Matches to DataFrame of all matched observations
    
    Parameters
    ----------
    matches : Match
        Match class object with matches already fit
    data : DataFrame 
        Dataframe with unique rows, for which we want to create new matched data.
        This may be a dataframe of covariates, treatment, outcome, or any combination.
    show_duplicates : bool
        Should repeated matches be included as multiple rows? Default is True.
        If False, then duplicates appear as one row but a column of weights is
        added.
    
    Returns
    -------
    DataFrame containing only the treatment group and matched controls,
    with the same columns as input data
    """
    
    if show_duplicates:
        indices = []
        for i in range(len(matches.freq)):
            j = matches.freq[i]
            while j>0:
                indices.append(i)
                j -= 1
        return data.loc[indices] # replace ix with loc
    else:
        dat2 = data.copy()
        dat2['weights'] = matches.weights
        dat2['frequency'] = matches.freq
        keep = dat2['frequency'] > 0
        return dat2.loc[keep]


def rank_test(covariates, groups):
    """ 
    Wilcoxon rank sum test for the distribution of treatment and control covariates.
    
    Parameters
    ----------
    covariates : DataFrame 
        Dataframe with one covariate per column.
        If matches are with replacement, then duplicates should be 
        included as additional rows.
    groups : array-like
        treatment assignments, must be 2 groups
    
    Returns
    -------
    A list of p-values, one for each column in covariates
    """    
    colnames = list(covariates.columns)
    J = len(colnames)
    pvalues = np.zeros(J)
    for j in range(J):
        var = covariates[colnames[j]]
        res = ranksums(var[groups == 1], var[groups == 0])
        pvalues[j] = res.pvalue
    return pvalues
    

def t_test(covariates, groups):
    """ 
    Two sample t test for the distribution of treatment and control covariates
    
    Parameters
    ----------
    covariates : DataFrame 
        Dataframe with one covariate per column.
        If matches are with replacement, then duplicates should be 
        included as additional rows.
    groups : array-like
        treatment assignments, must be 2 groups
    
    Returns
    -------
    A list of p-values, one for each column in covariates
    """
    colnames = list(covariates.columns)
    J = len(colnames)
    pvalues = np.zeros(J)
    for j in range(J):
        var = covariates[colnames[j]]
        res = ttest_ind(var[groups == 1], var[groups == 0],nan_policy='omit')
        pvalues[j] = res.pvalue
    return pvalues
    
def cohenD (tmp, metricName):
    treated_metric = tmp[tmp.treatment == 1][metricName]
    untreated_metric = tmp[tmp.treatment == 0][metricName]
    
    d = ( treated_metric.mean() - untreated_metric.mean() ) / math.sqrt(((treated_metric.count()-1)*treated_metric.std()**2 + (untreated_metric.count()-1)*untreated_metric.std()**2) / (treated_metric.count() + untreated_metric.count()-2))
    return d

def ks_2samp(covariates, groups):
    """ 
    Two sample t test for the distribution of treatment and control covariates
    
    Parameters
    ----------
    covariates : DataFrame 
        Dataframe with one covariate per column.
        If matches are with replacement, then duplicates should be 
        included as additional rows.
    groups : array-like
        treatment assignments, must be 2 groups
    
    Returns
    -------
    A list of p-values, one for each column in covariates
    """
    colnames = list(covariates.columns)
    J = len(colnames)
    pvalues = np.zeros(J)
    for j in range(J):
        var = covariates[colnames[j]]
        res = stat.ks_2samp(var[groups == 1], var[groups == 0],alternative='two-sided')
        pvalues[j] = res.pvalue
    return pvalues
    
def comparison_plot_hist(data, data_matched, feature, log_scale=False):
    plt.close()
    plt.figure()
    f, axes = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplot(121)
    sns.kdeplot(data = data[data['treatment_status']==1], x = feature,  ax = axes[0], log_scale = log_scale, label = 'treated').set()
    sns.kdeplot(data = data[data['treatment_status']==0], x = feature,  ax = axes[0], log_scale = log_scale, color='orange', label = 'untreated')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend(loc="upper left")
    plt.title('Before Matching')

    plt.subplot(122)
    sns.kdeplot(data = data_matched[data_matched['treatment_status']==1], x = feature,  ax = axes[1], log_scale = log_scale, label = 'treated').set()
    sns.kdeplot(data = data_matched[data_matched['treatment_status']==0], x = feature,  ax = axes[1], log_scale = log_scale, color='untreated', label = 'Did not consume MT')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title('After Matching')
    plt.legend(loc="upper left")
    plt.show()
    
def comparison_plot_cat(data, data_matched, feature):
    comparison_before = pd.DataFrame({'Treatment':data[data['treatment_status']==1][feature].value_counts(normalize=True),
                                      'Control':data[data['treatment_status']==0][feature].value_counts(normalize=True)})
    comparison_after = pd.DataFrame({'Treatment':data_matched[data_matched['treatment_status']==1][feature].value_counts(normalize=True),
                                     'Control':data_matched[data_matched['treatment_status']==0][feature].value_counts(normalize=True)})

 

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(30,5))
    comparison_before.sort_values('Treatment', ascending=False).plot(y=['Treatment', 'Control'], kind='bar', figsize=(15,5), alpha=0.5, ax=ax1)
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Density')
    ax1.set_title('Before Matching')
    comparison_after.sort_values('Treatment', ascending=False).plot(y=['Treatment', 'Control'], kind='bar', figsize=(15,5), alpha=0.5, ax=ax2)
    ax2.set_xlabel(feature)
    ax2.set_ylabel('Density')
    ax2.set_title('After Matching')
    plt.show()