import os, random
import numpy as np
import pandas as pd
import scipy.sparse as sp
calhost:8875
import anndata

try:
    import tensorflow as tf
except ImportError:
    raise ImportError('scAAnet requires tensorflow. Please follow instructions'
                      ' at https://www.tensorflow.org/install/ to install it.')

from .train import train
from .network import AE_types

def scAAnet(count,
        ae_type='zinb',
        hidden_size=(128, 10, 128), # network args
        hidden_dropout=0.,
        dispersion='gene-cell',
        batchnorm=True,
        activation='relu',
        init='glorot_normal',
        network_kwds={},
        epochs=300,               # training args
        reduce_lr=10,
        early_stop=15,
        batch_size=64,
        optimizer='rmsprop',
        learning_rate=0.01,
        random_state=0, 
        verbose=False,
        training_kwds={},
        return_model=False,
        return_loss=False,
        return_info=False,
        warm_up = 20
        ):
    """Single-Cell Archetypal Analysis Neural Network (scAAnet) API.

    Parameters
    ----------
    count : `anndata.AnnData`, `pandas.core.frame.DataFrame` or `numpy.ndarray`
        A dataframe saving raw counts.
    ae_type : `str`, optional. `zipoisson`(default), `zinb`, `nb` or `poisson`.
        Type of the autoencoder. Return values and the architecture is
        determined by the type e.g. `nb` does not provide dropout
        probabilities.
    hidden_size : `tuple` or `list`, optional. Default: (64, 32, 64).
        Width of hidden layers.
    hidden_dropout : `float`, `tuple` or `list`, optional. Default: 0.0.
        Probability of weight dropout in the autoencoder (per layer if list
        or tuple).
    dispersion : `str`, optional. `gene` (default) or `gene-cell`.
        If you want the dispersion parameter per gene to be the same across cells,
        leave it as default; otherwise, choosing `gene-cell` will make it differ
        across cells.
    batchnorm : `bool`, optional. Default: `True`.
        If true, batch normalization is performed.
    activation : `str`, optional. Default: `relu`.
        Activation function of hidden layers.
    init : `str`, optional. Default: `glorot_uniform`.
        Initialization method used to initialize weights.
    network_kwds : `dict`, optional.
        Additional keyword arguments for the autoencoder.
    epochs : `int`, optional. Default: 300.
        Number of total epochs in training.
    reduce_lr : `int`, optional. Default: 10.
        Reduces learning rate if validation loss does not improve in given number of epochs.
    early_stop : `int`, optional. Default: 15.
        Stops training if validation loss does not improve in given number of epochs.
    batch_size : `int`, optional. Default: 32.
        Number of samples in the batch used for SGD.
    optimizer : `str`, optional. Default: "rmsprop".
        Type of optimization method used for training.
    learning_rate : `float`, optional. Default: 0.01.
        Learning rate to use in the training.
    random_state : `int`, optional. Default: 0.
        Seed for python, numpy and tensorflow.
    verbose : `bool`, optional. Default: `False`.
        If true, prints additional information about training and architecture.
    training_kwds : `dict`, optional.
        Additional keyword arguments for the training process.
    return_model : `bool`, optional. Default: `False`.
        If true, trained autoencoder object is returned. See "Returns".
    return_loss : `bool`, optional. Default: `False`.
        If true, loss information is returned. See "Returns".
    return_info : `bool`, optional. Default: `False`.
        If true, all additional parameters of scAAnet are returned such as dropout
        probabilities and estimated dispersion values, in case that autoencoder 
        is of type zinb or zinb-conddisp.
    warm_up : `int`. Default: 5.
        Number of epochs for warm-up. During warm-ups, the weights of Z_fixed layer
        is not trained.

    Returns
    -------
    If `return_model` is true, reconstruction, usage, archetype matrices and a trained model 
    are returned. Otherwise, only the three matrices are returned as a tuple.
    If `return_loss` is true, two loss values are returned, including reconstruction loss (neg
    log-likelihood) and reconstruction + archetypal loss.
    """

    # set seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    os.environ['PYTHONHASHSEED'] = '0'
    
    if isinstance(count, anndata.AnnData):
        if sp.issparse(count.X):
            count = count.X.todense()
        else:
            count = count.X
    count = np.asmatrix(count).astype('float32')

    TPM = count/count.sum(axis=1)
    lib_size = count.sum(axis=1)
    
    network_kwds = {**network_kwds,
        'hidden_size': hidden_size,
        'hidden_dropout': hidden_dropout,
        'dispersion': dispersion,
        'batchnorm': batchnorm,
        'activation': activation,
        'init': init
    }

    input_size = output_size = count.shape[1]
    net = AE_types[ae_type](input_size=input_size,
                            output_size=output_size,
                            **network_kwds)

    net.save()
    net.build_enc()
    net.build_dec()

    training_kwds = {**training_kwds,
        'epochs': epochs,
        'reduce_lr': reduce_lr,
        /'early_stop': early_stop,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'verbose': verbose,
        'learning_rate': learning_rate,
        'ae_type': ae_type,
        'warm_up': warm_up,
        'Z_fixed_idx': len(hidden_size) + 1
    }

    train(count, TPM, lib_size, net, **training_kwds)
    preds = net.predict(TPM, lib_size, return_info)
    
    outputs = net.model([TPM, lib_size], training=False)
    loss_value = net.loss(count, outputs)
    loss_value_all = loss_value + sum(net.model.losses)
    losses = {"loss": loss_value.numpy(), "loss_all": loss_value_all.numpy()}
    if return_model:
        if return_loss:
            return preds, losses, net
        else:
            return preds, net
    else:
        if return_loss:
            return preds, losses
        else:
            return preds
