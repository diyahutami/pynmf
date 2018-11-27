"""
PyNMF Versatile Sparse Non-negative Matrix Factorization.

    VSNMF: Vesatile Sparse for Non-negative Matrix Factorization

[1] C.Ding, T.Li, W.Peng, and H.Park, "Orthogonal nonnegative matrix t-factorization for clustering",
in Proc. 12th ACM SIGKDD Int. Conf. Know. Discovery Data Mining, 2006, PP.126-135
[2] Choi, S.: Algorithms for orthogonal nonnegative matrix factorization. 
In: Proceedings of the International Joint Conference on Neural Networks (IJCNN), Hong Kong (2008)
"""
import numpy as np
import logging
import logging.config
import scipy.sparse
from .base import PyNMFBase
import numpy.matlib
import numpy.linalg

__all__ = ["ORTHOGONAL"]


class ORTHOGONAL(PyNMFBase):
    """
    ORTHOGONAL(data, num_bases=4, niter=10, orthogonal='A')

    Non-negative Matrix Factorization. Factorize a data matrix into two matrices
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the classicial multiplicative update rule.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)
    orthogonal : 'A' or 'Y'
        the input orthogonal to set the one-side orthogonal        

    """

    def __init__(self, data, num_bases=4, niter=10, orthogonal='A', **kwargs):
        PyNMFBase.__init__(self, data, num_bases, niter, **kwargs)
        self.orthogonal = orthogonal
    
    def _update_w(self):
        # pre init W1, and W2 (necessary for storing matrices on disk)
        if (self.orthogonal == 'A'):
            W1 = np.dot(np.dot(np.dot(self.W, self.W.T), self.data[:,:]), self.H.T) + 10**-9
            W2= np.sqrt(np.dot(self.data[:,:], self.H.T)/W1)
            self.W *= W2
            
        elif (self.orthogonal == 'Y'):
            W2 = np.dot(np.dot(self.W, self.H), self.H.T) + 10**-9
            self.W *= np.dot(self.data[:,:], self.H.T)
            self.W /= W2
            # to normalize
            # self.W /= np.sqrt(np.sum(self.W**2.0, axis=0))

        else:
            print("please insert side of orthogonal")

    def _update_h(self):
        # pre init H1, and H2 (necessary for storing matrices on disk)
        if (self.orthogonal == 'Y'):
            H1 = np.dot(np.dot(np.dot(self.H, self.data[:,:].T), self.W), self.H) + 10**-9
            H2= np.sqrt(np.dot(self.W.T, self.data[:,:])/H1)
            self.H *= H2

        elif (self.orthogonal == 'A'):
            H2 = np.dot(np.dot(self.W.T, self.W), self.H) + 10**-9
            self.H *= np.dot(self.W.T, self.data[:,:])
            self.H /= H2
            # to normalize
            # self.H /= np.sqrt(np.sum(self.H**2.0, axis=0))
            
        else:
            print("please insert side of orthogonal")



def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
