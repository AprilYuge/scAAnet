# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:06:57 2021

@author: wangy
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf 
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

import scipy.stats as stats

def save_df_to_npz(obj, filename):
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)

def load_df_from_npz(filename):
    with np.load(filename, allow_pickle = True) as f:
        obj = pd.DataFrame(**f)
    return obj

# Program specific DEG
def programDEG(count, usage, test_use = 'nb', offset = True, p_cor = 'bonferroni'):
    """

    Parameters
    ----------
    count : `pandas.core.frame.DataFrame`
        A dataframe saving raw counts with rows being cells and columns being genes.
    usage: `numpy.ndarray`
        A numpy array saving usage and each column corresponds to a program.
    test.use: `str`, optional. Default: `nb`.
        This is the type of GLM regression used for DEG, another choice is `poisson`
    p_cor: `str`, optional. Default: `bonferroni`.
        Method used for p value adjustment. Another choice is `benjamini-hochberg`.

    Returns
    -------
    A dictionary with each element corresponds to results for a program. Columns of each 
    dataframe are coefficient estimates, p values, and adjusted p values. 
    """
    
    results = {}
    
    lib_size = count.sum(1)
    
    # Filter out cells with too small library size
    min_counts_per_cell=1/20 * count.shape[1]
    count = count[lib_size >= min_counts_per_cell]
    usage = usage[lib_size >= min_counts_per_cell,:]
    lib_size = lib_size[lib_size >= min_counts_per_cell]
    
    for i in range(usage.shape[1]):
        
        # Save results for current program in result
        result = []
        count_a = 0

        for j in range(count.shape[1]):
            
            # Check if the current gene has 0 variation across cells
            if np.sum(count.iloc[:,j]>0)<=count.shape[0]/500:
                continue
            
            # Negatve binomial regression for the ith program and jth gene
            data = pd.DataFrame({'usage': usage[:,i], 'gene': count.iloc[:,j], 'log_lib_size': np.log(lib_size)})
            
            if offset:
                offs = data['log_lib_size']
                formula = "gene ~ usage"
            else:
                offs = None
                formula = "gene ~ usage + log_lib_size"
            
            if test_use == 'poisson':
                model = smf.glm(formula=formula, data=data, offset=offs, family=sm.families.Poisson()).fit()
            elif test_use == 'nb':      
                # EStimate alpha
                model_p = smf.glm(formula=formula, data=data, offset=offs, family=sm.families.Poisson()).fit()
                data['bb_lambda'] = model_p.mu
                data['aux_ols_dep'] = data.apply(lambda x: ((x['gene'] - x['bb_lambda'])**2 - x['bb_lambda']) / x['bb_lambda'], axis=1)
                model_aux_ols = smf.ols(formula="aux_ols_dep ~ bb_lambda - 1", data=data).fit()
            
                # print alpha
                #print(model_aux_ols.params[0])
                if model_aux_ols.params[0] < 0:
                    count_a += 1
                    
                # Truncate alpha
                alpha = 2 if model_aux_ols.params[0] > 2 else model_aux_ols.params[0]
                alpha = 0.01 if alpha < 0.01 else alpha
            
                # Fit NB regression model
                model = smf.glm(formula=formula, data=data, offset=offs, 
                                family=sm.families.NegativeBinomial(alpha=alpha)).fit()
            elif test_use == 'nb_naive':
                # Fit NB regression model
                model = smf.glm(formula=formula, data=data, offset=offs, 
                                family=sm.families.NegativeBinomial()).fit()

            # Append results
            result.append([count.columns[j], model.params[1], model.pvalues[1], model.tvalues[1]])
            
        result = pd.DataFrame(result, columns=['gene', 'coef.est', 'p.val', 'z.score'])
        result.loc[np.isnan(result['p.val']),'p.val'] = 1
        
        # Calculate adjusted p vals
        if p_cor == 'benjamini-hochberg':
            _, p_vals_adj, _, _ = multipletests(result['p.val'], alpha=0.05, method='fdr_bh')
            result['p.val.adj'] = p_vals_adj
        elif p_cor == 'bonferroni':
            result['p.val.adj'] = np.minimum(result['p.val']*result.shape[0], 1.0)
     
        # Add result
        results['gep_%s'%(i+1)] = result    
        
        if test_use == 'nb':
            print('Number of negative alpha is %s' %count_a)
    
    return results

# Ranked gene scores in each GEP
def plotRankedGeneScore(spectra, ncol=4):
    nrow = int(np.ceil(spectra.shape[1]/ncol))
    fig = plt.figure(figsize=(ncol*2, nrow*2), dpi=800)
    gs = gridspec.GridSpec(nrow, ncol, fig, 0, 0, 1, 1, hspace=.8)
    for (i,program) in enumerate(spectra.columns):
        ax = fig.add_subplot(gs[int(i/ncol), i%ncol], frameon=False, rasterized=False)
        x_ind = np.arange(1, spectra.shape[0]+1)
        spectra_sub = spectra.sort_values(by=spectra.columns[i], axis=0, ascending=False).iloc[:,i]
        value = spectra_sub.to_numpy()
        ax.scatter(x_ind, value, label=None, s=5, alpha=.8)
        #ax.set_xlabel('Ranked gene list')
        #ax.set_ylabel('Relative expression')
        ax.set_title('GEP %s'%program)
        
# Density plot of gene scores in each GEP
def plotDensity(spectra, ncol=4):
    nrow = int(np.ceil(spectra.shape[1]/ncol))
    fig = plt.figure(figsize=(ncol*2, nrow*2), dpi=800)
    gs = gridspec.GridSpec(nrow, ncol, fig, 0, 0, 1, 1, hspace=.8)
    for (i,program) in enumerate(spectra.columns):
        ax = fig.add_subplot(gs[int(i/ncol), i%ncol], frameon=False, rasterized=False)
        spectra_sub = spectra.iloc[:,i]
        value = spectra_sub.to_numpy()
        sns.distplot(value, hist=True, bins=30, ax=ax)
        ax.set_title('GEP %s'%program)
        
# Density plot of gene scores fitted by Gamma distributions in each GEP
def plotDensityGamma(spectra, ncol=4, thrP=0.975):
    selected_genes = {}
    nrow = int(np.ceil(spectra.shape[1]/ncol))
    fig = plt.figure(figsize=(ncol*2, nrow*2), dpi=800)
    gs = gridspec.GridSpec(nrow, ncol, fig, 0, 0, 1, 1, hspace=.8)
    for (i,program) in enumerate(spectra.columns):
        ax = fig.add_subplot(gs[int(i/ncol), i%ncol], frameon=False, rasterized=False)
        spectra_sub = spectra.iloc[:,i]
        value = spectra_sub.to_numpy()
        value[value==0] += 1e-8
        x = np.linspace(1e-3, value.max(), 100)
        # Fit a Gamma distribution
        gamma = stats.gamma
        param = gamma.fit(value, floc=0)
        pdf_fitted = gamma.pdf(x, *param)
        ax.plot(x, pdf_fitted, color='r')
        # Get gene score cutoff
        cutoff = gamma.ppf(thrP, *param)
        ax.axvline(cutoff, color='grey', linestyle='--')
        # Histogram
        ax.hist(value, bins=30, density=True)
        ax.set_title('GEP %s'%program)
        # Number of genes pass the cutoff
        num = np.sum(value > cutoff)
        ax.set_xlabel('%d genes with score > %.3f' %(num, cutoff))
        # Add genes to dictionary
        spectra_sub = spectra.sort_values(by=spectra.columns[i], axis=0, ascending=False).iloc[:,i]
        selected_genes['GEP %d'%(i+1)] = spectra_sub.index[:num]
    
    return selected_genes

# Calculate gene scores based on the equation given in cisTopic
def geneScore(spectra, scale=True):
    gene_score = spectra*(np.log(spectra+1e-5).sub(np.mean(np.log(spectra+1e-5), axis=1), axis='index'))
    if scale:
        gene_score = gene_score.sub(gene_score.min(axis=0), axis='columns').div(gene_score.max(axis=0)-gene_score.min(axis=0), 
                                                                                axis='columns')
    return gene_score