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

import scipy
from scipy.cluster.hierarchy import cophenet, leaves_list
from scipy.spatial.distance import squareform, pdist
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import fastcluster as fc


def save_df_to_npz(obj, filename):
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)

def load_df_from_npz(filename):
    with np.load(filename, allow_pickle = True) as f:
        obj = pd.DataFrame(**f)
    return obj

# Program specific DEG
def programDEG(count, usage, test_use = 'nb', offset = True, p_cor = 'bonferroni', lib_size=None, min_counts_per_cell=None):
    """

    Parameters
    ----------
    count : `pandas.core.frame.DataFrame`
        A dataframe saving raw counts with rows being cells and columns being genes.
    usage: `numpy.ndarray`
        A numpy array saving usage and each column corresponds to a program.
    test.use: `str`, optional. Default: `nb`.
        This is the type of GLM regression used for DEG, other choices are `nb_naive` and `poisson`.
    p_cor: `str`, optional. Default: `bonferroni`.
        Method used for p value adjustment. Another choice is `benjamini-hochberg`.

    Returns
    -------
    A dictionary with each element corresponds to results for a program. Columns of each 
    dataframe are coefficient estimates, p values, and adjusted p values. 
    """
    
    results = {}
    
    if lib_size is None:
        lib_size = count.sum(1)
    
    # Filter out cells with too small library size
    if min_counts_per_cell is None:
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

# Assess the stability of inferred archetypes through differnt Ks
def StabilityArchetype(k_min, k_max, step, n_rep, dists, filepath, savepath=None, l2 = False):
    
    '''
    Borrowed from Alexandrov et al. 2013: Deciphering signatures of mutational processes 
    operative in human cancer and Kotliar et al. 2019: Identifying gene expression programs
    of cell-type identify and cellular activity with single-cell RNA-seq
    '''
    
    K = np.arange(k_min, k_max+1, step)
    results = []
    
    for k in K:
        for dist in dists:
            # Read all spectra results
            arch = load_df_from_npz(filepath%(dist, 1, k))
            arch = arch.transpose()
            for rs in np.arange(2, n_rep+1):
                arch_new = load_df_from_npz(filepath%(dist, rs, k))
                arch = pd.concat([arch, arch_new.transpose()])
    
            if l2:
                arch = (arch.T/np.sqrt((arch**2).sum(axis=1))).T
                
            if savepath:
                save_df_to_npz(arch, savepath%(dist, k))
        
            # K-means clustering on all spectra
            kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
            kmeans_model.fit(arch)
            kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=arch.index)

            # Compute the silhouette score
            stability = silhouette_score(arch.values, kmeans_cluster_labels, metric='euclidean')
            results.append([k, dist, stability])
    
    results = pd.DataFrame(results, columns=["K", "Method", "Stability"])
    
    return results

# Assess the stability of inferred archetypes through differnt Ks
def StabilityUsage(k_min, k_max, step, n_rep, dists, n_cells, filepath_usage, filepath_consensus=None, 
                   n_cells_max=20000, cluster=True):
    
    '''
    Borrowed from Brunet et al. 2004: Metagenes and molecular pattern discovery using
    matrix factorization
    '''
    np.random.seed(520)
    
    K = np.arange(k_min, k_max+1, step)
    results = []
    
    if n_cells > n_cells_max:
        cells_choice = np.random.choice(n_cells, n_cells_max, replace=False)
        n_cells = n_cells_max
    else:
        cells_choice = np.arange(n_cells)
    
    for k in K:
        for dist in dists:
            d = np.zeros(int(scipy.special.comb(n_cells, 2)))
            # Read all cell usage results
            for rs in np.arange(1, n_rep+1):
                usage = load_df_from_npz(filepath_usage%(dist, rs, k))
                usage = usage.to_numpy()
                usage = usage[cells_choice,]               
                if cluster:
                    assign = usage.argmax(1)
                    usage = np.zeros_like(usage)
                    usage[np.arange(len(usage)), assign] = 1
                    d += scipy.spatial.distance.pdist(usage, 'braycurtis')
                else:
                    d += scipy.spatial.distance.pdist(usage)
            
            d = d/n_rep
            
            if filepath_consensus:
                if cluster:
                    # Save consensus clustering matrix
                    np.save(filepath_consensus%(dist, k), 1-d)
                else:
                    # Save distance matrix
                    np.save(filepath_consensus%(dist, k), d)
            
            # Hierarchical clustering using distance d
            HC = fc.linkage(d, method='average')
            cophen_d = cophenet(HC)
            
            # Compute Cophenetic correlation coefficient
            cophen_corr = np.corrcoef(d, cophen_d)[0,1]
            results.append([k, dist, cophen_corr])
    
    results = pd.DataFrame(results, columns=["K", "Method", "Stability"])
    
    return results

# Plot consensus clustering matrix of inferred usage
def PlotConsensusMat(filepath, savepath=None, colormap='rocket', n_cells_max=6000, dpi=600):
    np.random.seed(520)
    cvec = np.load(filepath)    
    cmat = squareform(cvec)
    if len(cmat) > n_cells_max:
        cells_choice = np.random.choice(len(cmat), n_cells_max, replace=False)
        cmat = cmat[cells_choice,][:, cells_choice]
        cvec = scipy.spatial.distance.squareform(cmat)
    HC = fc.linkage(1-cvec, method='average')
    cg = sns.clustermap(cmat, row_linkage=HC, col_linkage=HC, cbar_pos=(1, .2, .03, .4), 
                        yticklabels=False, xticklabels=False, cmap=colormap)
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    if savepath:
        plt.savefig(savepath, dpi=dpi)
        plt.close()

# Plot distance clustering matrix of inferred GEPs
def PlotArchDistMat(filepath, savepath=None, colormap='rocket', dpi=600):
    arch = load_df_from_npz(filepath)
    
    # K-means clustering on all spectra
    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans_model.fit(arch)
    kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=arch.index)
    
    topics_dist = squareform(pdist(arch.values))
    
    spectra_order = []
    for cl in sorted(set(kmeans_cluster_labels)):
        cl_filter = kmeans_cluster_labels==cl

        if cl_filter.sum() > 1:
            cl_dist = squareform(topics_dist[cl_filter, :][:, cl_filter])
            cl_dist[cl_dist < 0] = 0 #Rarely get floating point arithmetic issues
            cl_link = fc.linkage(cl_dist, 'average')
            cl_leaves_order = leaves_list(cl_link)

            spectra_order += list(np.where(cl_filter)[0][cl_leaves_order])
        else:
            ## Corner case where a component only has one element
            spectra_order += list(np.where(cl_filter)[0])
            
    width_ratios = [0.5, 9, 0.5, 0.5]
    height_ratios = [0.5, 9]
    fig = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
    gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), fig,
                            0.01, 0.01, 0.98, 0.98,
                            height_ratios=height_ratios,
                            width_ratios=width_ratios,
                            wspace=0, hspace=0)

    dist_ax = fig.add_subplot(gs[1,1], xticks=[], yticks=[], frameon=False)

    D = topics_dist[spectra_order, :][:, spectra_order]
    dist_im = dist_ax.imshow(D, interpolation='none', cmap=colormap, aspect='auto', rasterized=True)

    left_ax = fig.add_subplot(gs[1,0], xticks=[], yticks=[], frameon=False)
    left_ax.imshow(kmeans_cluster_labels.values[spectra_order].reshape(-1, 1),
                    interpolation='none', cmap='Spectral', aspect='auto', rasterized=True)

    top_ax = fig.add_subplot(gs[0,1], xticks=[], yticks=[], frameon=False)
    top_ax.imshow(kmeans_cluster_labels.values[spectra_order].reshape(1, -1),
                  interpolation='none', cmap='Spectral', aspect='auto', rasterized=True)

    cbar_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 3],
                                               height_ratios=[0.5,1,0.5], wspace=0, hspace=0)
    cbar_ax = fig.add_subplot(cbar_gs[1,0], frameon=False, title='')
    fig.colorbar(dist_im, cax=cbar_ax, ticks=np.linspace(D.min(), D.max(), 6).round(2), 
                 orientation='vertical').outline.set_visible(False)
    
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches = 'tight')
        plt.close()
