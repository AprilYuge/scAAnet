# Single-cell archetypal analysis neural network (scAAnet) #

scAAnet is an autoencoder-based neural network which can perform nonlinear archetypal analysis on single-cell RNA-seq data. The underlying assumption is that the expression profile of each single cell is a nonlinear combination of several gene expression programs [1] (GEPs) with each program corresponds to an archetype in the archetypal analysis. The purpose of scAAnet is to decompose an expression profile into a usage matrix and a GEP/archetype matrix. From the usage matrix, we can know how much a cell utilizes different GEPs and from the GEP matrix we will be able to quantify the relative importance of genes in each GEP. One novelty of this method is to make use of the nonlinearity of neural network to take into consideration the complex interaction among genes. Another novelty is that the negative log-likelihood of some discrete distribution (Poisson, zero-inflated Poisson, negative binomial, or zero-inflated negative binomial distribution) is used as the reconstruction loss instead of the traditional MSE loss. This is because single-cell RNA-seq data is a type of count data. 

![alt text](https://github.com/AprilYuge/scAAnet_latest/blob/main/images/overview.png)

More details about scAAnet can be found in our [manuscript](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010025).

### Usage ###

scAAnet is implemented in Python. To use scAAnet, TensorFlow 2 is required. A [virtual environment](https://docs.python.org/3/tutorial/venv.html) is recommended to be used for installing these packages. You can create a virtual environment and install packages as follows:

* Create a virtual environment by `python3 -m venv ./venv` (./venv can be changed to your path).
* Activate the virtual environment by `source ./venv/bin/activate`.
* Install [tensorflow](https://www.tensorflow.org/install) by `pip install --upgrade tensorflow`.
* Install scAAnet by `pip install scAAnet`.

Then you can use scAAnet after activating the virtual environment. Here is an example of running scAAnet:

* `from scAAnet.api import scAAnet`
* `re = scAAnet(count, hidden_size=(128, K, 128), ae_type='zinb', epochs=200, batch_size=64, early_stop=100, reduce_lr=10, learning_rate=0.01)`
* `recon, usage, spectra = re['recon'], re['usage'], re['spectra']`

The input `count` variable is single-cell expression raw count data with N cells and G genes and `K` is the number of archetypes/GEPs. The input `count` can be a pandas dataframe, a numpy array, or an AnnData object. Note that `recon`, `usage` and `spectra` are reconstructed expression count data (N by G), the usage matrix (N by K) and the archetype matrix (K by G) of the input count data, respectively. The argument ae_type can be chosen from poisson, zipoisson, nb and zinb.

More details about how to use scAAnet can be found in this [tutorial](https://github.com/AprilYuge/scAAnet_latest/blob/main/tutorials/Tutorial%20of%20scAAnet%20on%20simulated%20data.ipynb) on simulated data based on Splatter’s [2] framework. Analysis code for the manuscript and more usage of scAAnet can be found in this [folder](https://github.com/AprilYuge/scAAnet_latest/tree/main/scripts). Packages recommended for installation for downstream analysis accompanying scAAnet are [Scanpy](https://scanpy.readthedocs.io/en/stable/installation.html) and [GSEAPY](https://gseapy.readthedocs.io/en/latest/introduction.html). Scanpy is a widely-used toolkit for scalable single-cell analysis in Python and GSEAPY is a Python wrapper for [GSEA](https://www.gsea-msigdb.org/gsea/index.jsp) and [Enrichr](https://maayanlab.cloud/Enrichr/) that is used for pathway enrichment analysis.

Once you are done with the analysis you can deactivate the virtual environment by typing `deactivate` in your shell.

### References ###

* [1]: Kotliar, Dylan, et al. "Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq." Elife 8 (2019).
* [2]: Zappia, Luke, Belinda Phipson, and Alicia Oshlack. "Splatter: simulation of single-cell RNA sequencing data." Genome biology 18.1 (2017): 174.


