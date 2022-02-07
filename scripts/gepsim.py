# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:42:47 2020

@author: wangy
"""

# This file includes a function used for simulating data. This was based on Splatter (Zappia, Phipson, and Oshlack, 2017) and adapated from code on https://github.com/dylkot/scsim.

import pandas as pd
import numpy as np

# Added de_overlap which means whether different groups can share DEGs.

class gepsim:
    def __init__(self, ngenes=10000, ncells=100, seed=757578,
                 mean_rate=.3, mean_shape=.6, libloc=11, libscale=0.2,
                expoutprob=.05, expoutloc=4, expoutscale=0.5, ngroups=1,
                diffexpprob=.1, diffexpdownprob=.5,
                diffexploc=.1, diffexpscale=.4, bcv_dispersion=.1,
                bcv_dof=60, ndoublets=0, boundprob=.5, 
                zeroinflate = False, zidecay = 0.3, de_overlap = True, cluster = False):
        
        self.ngenes = ngenes
        self.ncells = ncells
        self.seed = seed
        self.mean_rate = mean_rate
        self.mean_shape = mean_shape
        self.libloc = libloc
        self.libscale = libscale
        self.expoutprob = expoutprob
        self.expoutloc = expoutloc
        self.expoutscale = expoutscale
        self.ngroups = ngroups
        self.diffexpprob = diffexpprob
        self.diffexpdownprob = diffexpdownprob
        self.diffexploc = diffexploc
        self.diffexpscale = diffexpscale
        self.bcv_dispersion = bcv_dispersion
        self.bcv_dof = bcv_dof
        self.ndoublets = ndoublets
        self.init_ncells = ncells+ndoublets
        self.boundprob = boundprob
        self.zidecay = zidecay
        self.zeroinflate = zeroinflate
        self.de_overlap = de_overlap
        self.cluster = cluster

    def simulate(self):
        np.random.seed(self.seed)
        print('Simulating cells')
        self.cellparams = self.get_cell_params()
        print('Simulating gene params')
        self.geneparams = self.get_gene_params()

        print('Simulating DE')
        self.sim_group_DE()

        print('Simulating cell-gene means')
        self.cellgenemean = self.get_cell_gene_means()
        if self.ndoublets > 0:
            print('Simulating doublets')
            self.simulate_doublets()

        print('Adjusting means')
        self.adjust_means_bcv()
        print('Simulating counts')
        self.simulate_counts()
        if self.zeroinflate:
            print('Simulating zero inflated counts')
            self.simulate_zicounts()
        
        
    def sample_at_uniform(self, ngroups, n):
        ''' Method for uniformly sampling from a simplex'''
        u = np.random.uniform(0,1,[n, ngroups])
        e = -np.log(u)
        x = e / np.sum(e, axis=1, keepdims=True)
        return x


    def sample_boundary_uniform(self, n):
        '''Sample the points along the boundary of a simplex'''
        x_all = []
        groupid = np.random.choice(np.arange(1, self.ngroups+1),
                                       size=n, p=[1/float(self.ngroups)]*self.ngroups)
        for i in range(self.ngroups):
            x = self.sample_at_uniform(self.ngroups-1, (groupid==(i+1)).sum())
            x = np.insert(x, i, 0, axis=1)
            x_all.append(x)
        return np.concatenate(x_all, axis=0)


    def simulate_counts(self):
        '''Sample read counts for each gene x cell from Poisson distribution
        using the variance-trend adjusted updatedmean value'''
        self.counts = pd.DataFrame(np.random.poisson(lam=self.updatedmean),
                                   index=self.cellnames, columns=self.genenames)
        
    def simulate_zicounts(self):
        '''Add dropouts to read counts for each gene x cell'''
        # self.zicounts = np.where(np.random.binomial(1,np.exp(-self.zidecay*self.counts**2)) == 1, 
        #                          0, self.counts)
        self.zicounts = np.where(np.random.binomial(1,np.exp(-self.zidecay*self.updatedmean**2)) == 1, 
                                0, self.counts)
        self.zicounts = pd.DataFrame(self.zicounts, index=self.cellnames, columns=self.genenames)


    def adjust_means_bcv(self):
        '''Adjust cellgenemean to follow a mean-variance trend relationship'''
        self.bcv = self.bcv_dispersion + (1 / np.sqrt(self.cellgenemean))
        chisamp = np.random.chisquare(self.bcv_dof, size=self.ngenes)
        self.bcv = self.bcv*np.sqrt(self.bcv_dof / chisamp)
        self.updatedmean = np.random.gamma(shape=1/(self.bcv**2),
                                           scale=self.cellgenemean*(self.bcv**2))
        self.bcv = pd.DataFrame(self.bcv, index=self.cellnames, columns=self.genenames)
        self.updatedmean = pd.DataFrame(self.updatedmean, index=self.cellnames,
                                        columns=self.genenames)
        
        # bcv = bcv_dispersion + (1 / np.sqrt(cellgenemean))
        # chisamp = np.random.chisquare(bcv_dof, size=ngenes)
        # bcv = bcv*np.sqrt(bcv_dof / chisamp)
        # updatedmean = np.random.gamma(shape=1/(bcv**2),
        #                                    scale=cellgenemean*(bcv**2))
        # updatedmean = pd.DataFrame(updatedmean, index=count.index, columns=count.columns)
        # counts = pd.DataFrame(np.random.poisson(lam=updatedmean),
        #                       index=count.index, columns=count.columns)


    def simulate_doublets(self):
        ## Select doublet cells and determine the second cell to merge with
        d_ind = sorted(np.random.choice(self.ncells, self.ndoublets,
                                        replace=False))
        d_ind = ['Cell%d' % (x+1) for x in d_ind]
        self.cellparams['is_doublet'] = False
        self.cellparams.loc[d_ind, 'is_doublet'] = True
        extraind = self.cellparams.index[-self.ndoublets:]
        group2 = self.cellparams.ix[extraind, 'group'].values
        self.cellparams['group2'] = -1
        self.cellparams.loc[d_ind, 'group2'] = group2

        ## update the cell-gene means for the doublets while preserving the
        ## same library size
        dmean = self.cellgenemean.loc[d_ind,:].values
        dmultiplier = .5 / dmean.sum(axis=1)
        dmean = np.multiply(dmean, dmultiplier[:, np.newaxis])
        omean = self.cellgenemean.loc[extraind,:].values
        omultiplier = .5 / omean.sum(axis=1)
        omean = np.multiply(omean, omultiplier[:,np.newaxis])
        newmean = dmean + omean
        libsize = self.cellparams.loc[d_ind, 'libsize'].values
        newmean = np.multiply(newmean, libsize[:,np.newaxis])
        self.cellgenemean.loc[d_ind,:] = newmean
        ## remove extra doublet cells from the data structures
        self.cellgenemean.drop(extraind, axis=0, inplace=True)
        self.cellparams.drop(extraind, axis=0, inplace=True)
        self.cellnames = self.cellnames[0:self.ncells]


    def get_cell_gene_means(self):
        '''Calculate each gene's mean expression for each cell while adjusting
        for the library size'''

        group_genemean = self.geneparams.loc[:,[x for x in self.geneparams.columns if ('_genemean' in x) and ('group' in x)]].T.astype(float)
        group_genemean = group_genemean.div(group_genemean.sum(axis=1), axis=0)
        group_usage = self.cellparams.loc[:,[x for x in self.cellparams.columns if ('_usage' in x) and ('group' in x)]].astype(float)

        cellgenemean = group_usage.to_numpy().dot(group_genemean.to_numpy())
        cellgenemean = pd.DataFrame(cellgenemean, index=group_usage.index, columns=group_genemean.columns)

        print('   - Normalizing by cell libsize')
        normfac = (self.cellparams['libsize'] / cellgenemean.sum(axis=1)).values
        for col in cellgenemean.columns:
            cellgenemean[col] = cellgenemean[col].values*normfac
        #cellgenemean = cellgenemean.multiply(normfac, axis=0).astype(float)
        return(cellgenemean)

    def get_gene_params(self):
        '''Sample each genes mean expression from a gamma distribution as
        well as identifying outlier genes with expression drawn from a
        log-normal distribution'''
        basegenemean = np.random.gamma(shape=self.mean_shape,
                                       scale=1./self.mean_rate,
                                       size=self.ngenes)

        is_outlier = np.random.choice([True, False], size=self.ngenes,
                                      p=[self.expoutprob,1-self.expoutprob])
        outlier_ratio = np.ones(shape=self.ngenes)
        outliers = np.random.lognormal(mean=self.expoutloc,
                                       sigma=self.expoutscale,
                                       size=is_outlier.sum())
        outlier_ratio[is_outlier] = outliers
        gene_mean = basegenemean.copy()
        median = np.median(basegenemean)
        gene_mean[is_outlier] = outliers*median
        self.genenames = ['Gene%d' % i for i in range(1, self.ngenes+1)]
        geneparams = pd.DataFrame([basegenemean, is_outlier, outlier_ratio, gene_mean],
                                  index=['BaseGeneMean', 'is_outlier', 'outlier_ratio', 'gene_mean'],
                                 columns=self.genenames).T
        return(geneparams)


    def get_cell_params(self):
        '''Sample cell library sizes'''
        libsize = np.random.lognormal(mean=self.libloc, sigma=self.libscale,
                                      size=self.init_ncells)
        self.cellnames = ['Cell%d' % i for i in range(1, self.init_ncells+1)]
        if self.cluster:
            '''Sample cluster label for each cell'''
            usage = np.zeros((self.ncells, self.ngroups))
            col = np.random.randint(0, self.ngroups, self.ncells)
            usage[np.arange(self.ncells), col] = 1
        else:
            '''Sample group usage for each cell'''
            nbound = int(self.ncells*self.boundprob)
            usage = np.concatenate([self.sample_at_uniform(self.ngroups, self.ncells-nbound), 
                                    self.sample_boundary_uniform(nbound)], axis=0)
        cellparams = pd.DataFrame(usage, index=self.cellnames, 
                                  columns=['group%d_usage' % i for i in range(1, self.ngroups+1)])
        cellparams['libsize'] = libsize
        return(cellparams)


    def sim_group_DE(self):
        '''Sample differentially expressed genes and the DE factor for each group'''
        self.groups = np.arange(1, self.ngroups+1)
        
        isDE_old = np.repeat(False, self.ngenes)
        for group in self.groups:
            if self.de_overlap == True:
                isDE = np.random.choice([True, False], size=self.ngenes,
                                        p=[self.diffexpprob,1-self.diffexpprob])
            else:
                diffexpprob = self.diffexpprob*self.ngenes/np.sum(np.invert(isDE_old))
                isDE = np.repeat(False, self.ngenes)
                isDE[np.invert(isDE_old)] = np.random.choice([True, False], size=self.ngenes - np.sum(isDE_old),
                                                              p=[diffexpprob,1-diffexpprob])
            DEratio = np.random.lognormal(mean=self.diffexploc,
                                          sigma=self.diffexpscale,
                                          size=isDE.sum())
            DEratio[DEratio<1] = 1 / DEratio[DEratio<1]
            is_downregulated = np.random.choice([True, False],
                                            size=len(DEratio),
                                            p=[self.diffexpdownprob,1-self.diffexpdownprob])
            DEratio[is_downregulated] = 1. / DEratio[is_downregulated]
            all_DE_ratio = np.ones(self.ngenes)
            all_DE_ratio[isDE] = DEratio
            group_mean = self.geneparams['gene_mean']*all_DE_ratio

            deratiocol = 'group%d_DEratio' % group
            groupmeancol = 'group%d_genemean' % group
            self.geneparams[deratiocol] = all_DE_ratio
            self.geneparams[groupmeancol] = group_mean
            isDE_old = isDE_old|isDE
