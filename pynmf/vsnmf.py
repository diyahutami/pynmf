"""
PyNMF Versatile Sparse Non-negative Matrix Factorization.

    VSNMF: Vesatile Sparse for Non-negative Matrix Factorization

[1] Y. Li and A. Ngom, "Versatile sparse matrix factorization and its applications in high-dimensional 
biological data", Pattern Recognition in Bioinformatics. Berlin, Germany: Springer, 2013, pp. 91-101
[2] Y. Li and A. Ngom, "Sparse representation approaches for the classification of high-dimensional 
biological data", BMC Syst. Biol., vol 7, no Suppl 4, 2013, Art. no S6

"""
import numpy as np
import logging
import logging.config
import scipy.sparse
from .base import PyNMFBase
import numpy.matlib
import numpy.linalg

__all__ = ["VSNMF"]


class VSNMF(PyNMFBase):
    """
    VSNMF(data, num_bases=4, niter=10, alfa1=0, alfa2=0, lambda1=0, lambda2=0, t1=1, t2=1)

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)       
    niter : int, optional
        Number of iteration
        10 (default)
    alfa1 : real value, optional (alfa1 >=0)
        Value of alfa1 to control the sparsity of the basis vectors
        0 (default)
    alfa2 : real value, optional (alfa2 >=0)
        Value of alfa2 to control the smoothness of the basis vectors
        0 (default)
    lambda1 : real value, optional (lambda1 >=0)
        Value of lambda1 to control the sparsity of the coefficeint vectors
        0 (default)
    lambda2 : real value, optional (lambda2 >=0)
        Value of lambda2 to control the smoothness of the coefficeint vectors
        0 (default)
    t1 : boolean value, optional ( 0 or 1)
        Value of t1 that indicate if non-negativity should or should not be enforced for W
        1 (default)
    t2 : boolean value, optional ( 0 or 1)
        Value of t2 that indicate if non-negativity should or should not be enforced for H
        1 (default)
         

    """

    def __init__(self, data, num_bases=4, niter=10, alfa1=0, alfa2=0, lambda1=0, lambda2=0, t1=1, t2=1, **kwargs):
        PyNMFBase.__init__(self, data, num_bases, niter, **kwargs)

        self.alfa1 = alfa1
        self.alfa2 = alfa2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.t1 = t1
        self.t2 = t2


    def frobenius_norm(self):
        # check if W and H exist
        if hasattr(self,'H') and hasattr(self,'W'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:,:] - (self.W * self.H)
                tmp = tmp.multiply(tmp).sum()
                a2 = 0.5 * self.alfa2 * np.trace(self.W[:,:].T * self.W[:,:])
                a1 = self.alfa1 * np.trace((np.matlib.repmat(np.array(1), self._data_dimension, self.num_bases).T) * self.W[:,:])
                l2 = 0.5 * self.lambda2 * np.trace(self.H[:,:].T * self.H[:,:])
                l1 = self.lambda1 * np.trace((np.matlib.repmat(np.array(1), self.num_bases, self._num_samples).T) * self.H[:,:])
                err = (0.5*tmp) + a2 + a1 + l2 + l1
                #err = (tmp) + a2 + a1 + l2 + l1
                #err = np.sqrt(err)
            else:
                tmp = np.sum((self.data[:,:] - np.dot(self.W, self.H))**2 ) 
                a2 = 0.5 * self.alfa2 * np.trace(np.dot(self.W[:,:].T, self.W[:,:]))
                a1 = self.alfa1 * np.trace(np.dot((np.matlib.repmat(np.array(1), self._data_dimension, self._num_bases).T), self.W[:,:]))
                l2 = 0.5 * self.lambda2 * np.trace(np.dot(self.H[:,:].T, self.H[:,:]))
                l1 = self.lambda1 * np.trace(np.dot((np.matlib.repmat(np.array(1), self._num_bases, self._num_samples).T), self.H[:,:]))
                err = (0.5*tmp) + a2 + a1 + l2 + l1
                #err = (tmp) + a2 + a1 + l2 + l1
                #err = np.sqrt(err)
        else:
            err = None

        return err

    
    def _update_w(self):
        # pre init W2 (necessary for storing matrices on disk)
        if (self.t1 == 1):
            W2 = np.dot(np.dot(self.W, self.H), self.H.T) + (self.alfa2 * self.W) + (self.alfa1 * (np.matlib.repmat(np.array(1), self._data_dimension, self._num_bases))) + 10**-9
            self.W *= np.dot(self.data[:,:], self.H.T)
            self.W /= W2
            # print('update w1')
            # to normalize
            # self.W /= np.sqrt(np.sum(self.W**2.0, axis=0))
        
        elif self.t1 == 0 and self.alfa1 == 0:
            W2 = np.linalg.inv(np.dot(self.H, self.H.T) + (self.alfa2 * np.identity(self._num_bases)))
            self.W = np.dot(np.dot(self.data[:,:], self.H.T), W2)

        elif self.t1 == 0:
            W2 = np.dot(np.dot(self.W, self.H), self.H.T) + (self.alfa2 * self.W) + (self.alfa1 * (np.matlib.repmat(np.array(1), self._data_dimension, self._num_bases))) + 10**-9
            self.W *= ( np.dot(self.data[:,:], self.H.T) + (self.alfa1 * (np.matlib.repmat(np.array(1), self._data_dimension, self._num_bases))) ) 
            self.W /= W2  
            # print('update w3')
            # to normalize
            # self.W /= np.sqrt(np.sum(self.W**2.0, axis=0))

        else:
            print("t1 value must be 0 or 1")
            
    def _update_h(self):
        # pre init H2 (necessary for storing matrices on disk)
        if (self.t2 == 1):
            H2 = np.dot(np.dot(self.W.T, self.W), self.H) + (self.lambda2 * self.H) + (self.lambda1 * (np.matlib.repmat(np.array(1), self._num_bases, self._num_samples))) + 10**-9
            self.H *= np.dot(self.W.T, self.data[:,:])
            self.H /= H2
            # print('update h1')
            # to normalize
            # self.H /= np.sqrt(np.sum(self.H**2.0, axis=0))
        
        elif self.t2 == 0 and self.lambda1 == 0:
            H2 = np.linalg.inv(np.dot(self.W.T, self.W) + (self.lambda2 * np.identity(self._num_bases)))
            self.H = np.dot(H2, np.dot(self.W.T, self.data[:,:]))

        elif self.t2 == 0:
            H2 = np.dot(np.dot(self.W.T, self.W), self.H) + (self.lambda2 * self.H) + (self.lambda1 * (np.matlib.repmat(np.array(1), self._num_bases, self._num_samples))) + 10**-9
            self.H *= ( np.dot(self.W.T, self.data[:,:]) + (self.lambda1 * (np.matlib.repmat(np.array(1), self._num_bases, self._num_samples))) )
            self.H /= H2
            # print('update h3')
            # to normalize
            # self.H /= np.sqrt(np.sum(self.H**2.0, axis=0))

        else:
            print("t2 value must be 0 or 1")


def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
