from os.path import dirname, abspath, join
from warnings import warn
import numpy as np
import random
from sklearn import datasets

data = datasets.load_iris()['data']

def dropout(a, percent):
    # create a copy
    mat = a.copy()
    # number of values to replace
    prop = int(mat.size * percent)
    # indices to mask
    mask = random.sample(range(mat.size), prop)
    # replace with NaN
    np.put(mat, mask, [np.NaN]*len(mask))
    return mat

def read():
    """
    Read ALL AML gene expression data. The matrix's shape is 5000 (genes) x 38 (samples).
    It contains only positive data.

    Return the gene expression data matrix.
    """
    fname = join(dirname(abspath(__file__)), 'datasets', 'ALL_AML', 'ALL_AML_data.txt')
    V = np.loadtxt(fname)
    return V

data = read()
modified = dropout(data, 0.3)
foutput = join(dirname(abspath(__file__)), 'datasets', 'ALL_AML', 'ALL_AML_data_30.txt')
np.savetxt(foutput, modified)