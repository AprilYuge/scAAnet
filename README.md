# Single-cell archetypal analysis neural network (scAAnet) #

scAAnet is an autoencoder-based neural network which can perform nonlinear archetypal analysis on single-cell RNA-seq data. The underlying assumption is that the expression profile of each single cell is a nonlinear combination of several gene expression programs [1] (GEPs) with each program corresponds to an archetype in the archetypal analysis. The purpose of scAAnet is to decompose an expression profile into a usage matrix and a GEP/archetype matrix. From the usage matrix, we can know how much a cell utilizes different GEPs and from the GEP matrix we will be able to quantify the relative importance of genes in each GEP. One novelty of this project is to make use of the nonlinearity of neural network to take into consideration the complex interaction among genes. Another novelty is that the negative log-likelihood of some discrete distribution (Poisson, zero-inflated Poisson, negative binomial, or zero-inflated negative binomial distribution) is used as the reconstruction loss instead of the traditional MSE loss. This is because single-cell RNA-seq data is a type of count data.

### Usage ###

scAAnet is implemented in Python. To use scAAnet, TensorFlow 2, numpy and pandas are required. A virtual environment is recommended to be used for installing these packages. You can create a virtual environment and install packages as follows:

* Create a virtual environment by `python3 -m venv ./venv` (./venv can be changed to your path).
* Activate the virtual environment by `source ./venv/bin/activate`.
* Install required packages by `pip install --upgrade tensorflow` (numpy wil be installed together with tensorflow) and `pip install pandas`.

Then you can use scAAnet after activating the virtual environment. Here is an example of running scAAnet:

* `from scAAnet.api import scAAnet`
* `recon, usage, spectra = scAAnet(count, threads=1, hidden_size=(128, 4, 128), ae_type='poisson', return_model=False, epochs=200, batch_size=64, early_stop=100, reduce_lr=10, learning_rate=0.01)`

Note that recon, usage and spectra are reconstructed expression count data, the usage matrix and the archetype matrix of the input count data, respectively. The argument ae_type can be chosen from poisson, zipoisson, nb and zinb. The input count variable should be a pandas dataframe with each row representing a cell and each column representing a gene.

### Simulation ###

Single-cell datasets were simulated based on Splatter’s [2] framework, but it was re-implemented in Python to allow the simulation of GEPs. The underlying distribution for expression count data is zero-inflated Negative-Binomial. The degree of zero-inflation is controlled by a parameter called zidecay. The default number of archetypes is 4 (K=4). The number of cells and the number of genes for each simulated dataset are 3,000 and 2,000 respectively. Data were simulated under three different signal-to-noise levels (the variable deloc in simulate.py) and for each level I generated 10 datasets with the same set of parameters using different random seeds. For each dataset, four original files were generated, cellparams.npz, geneparams.npz, truecounts.npz, and zicounts.npz. They contain cell parameters, gene parameters, gene expression counts without zero-inflation and counts with zero-inflation, respectively. All the parameters or variables can be modified in the simulate.py file under the simulation folder.

After simulation, each dataset was processed to filter cells and genes with low-quality. Cells with less than 100 Unique Molecular Identifier counts and genes that were not detected in at least 1 out of 500 cells were dropped. A file called countfiler.npz was saved for each dataset after processing.

Simulation described above can be conducted by running the simulate.py file under a directory where you want to save those simulated datasets.

### References ###

* [1]: Kotliar, Dylan, et al. "Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq." Elife 8 (2019).
* [2]: Zappia, Luke, Belinda Phipson, and Alicia Oshlack. "Splatter: simulation of single-cell RNA sequencing data." Genome biology 18.1 (2017): 174.

### Release notes ###

* 1.0.0 First release

### Change log ###

* 1.0.0 First release
* 0.15.1 Fixed some typos in simulate.py
* 0.15.0 Added some functions in tools.py as tools for simulation 
* 0.14.0 Defined a gepsim function for simulation
* 0.13.0 Writed a simulate.py to simulate scRNA-seq data with different signal-to-noise levels
* 0.12.1 Fixed some typos in network.py
* 0.12.0 Defined an api function as our scAAnet's API
* 0.11.0 Updated the train function to incorporate early stopping
* 0.10.0 Defined a train function for training
* 0.9.0 Defined class NBAutoencoder for ZINB
* 0.8.0 Defined class NBAutoencoder for NB
* 0.7.0 Defined class ZIPoissonAutoencoder for ZIP
* 0.6.0 Defined class PoissonAutoencoder for Poisson
* 0.5.0 Defined class Autoencoder as our baseline network structure
* 0.4.0 Defined a loss function for zero-inflated Negative-Binomial (ZINB)
* 0.3.0 Defined a loss function for Negative-Binomial (NB)
* 0.2.0 Defined a loss function for zero-inflated Poisson (ZIP)
* 0.1.0 Defined a loss function for Poisson
