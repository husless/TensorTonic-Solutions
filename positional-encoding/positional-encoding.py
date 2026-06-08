import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    nrows = seq_len
    ncols = d_model
    cc, pos = np.meshgrid(np.arange(ncols), np.arange(nrows))
    pe = np.zeros((nrows, ncols))
    pe[:, 0::2] = np.sin(pos[:, 0::2]/base**(cc[:,0::2]/d_model))
    pe[:, 1::2] = np.cos(pos[:, 1::2]/base**((cc[:,1::2]-1)/d_model))
    return pe