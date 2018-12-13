"""
PyNMF Orthogonal Non-negative Matrix Tri-Factorization.

    Bi0NMF: Orthogonal for Non-negative Matrix Tri-Factorization

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

__all__ = ["BIORTHOGONAL"]


class BIORTHOGONAL(PyNMFBase):
    """
    BIORTHOGONAL(data, num_bases=4, niter=10, orthogonal='AY')
    Orthogonal Non-negative Matrix Tri-factorizations for Clustering
    Orthogonal NMF factorize a data matrix into 3 matrices
    s.t. F = | data - A*S*Y | = | is minimal. A, S and Y are restricted to non-negativ data. 
    Uses the classicial multiplicative update rule.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)
    niter: int, optional
        Number of iteration
        10 (default)
    orthogonal : 'A' or 'Y' or 'AY'
        A : enforce orthogonal constraint on A (one-side orthogonal) 
        Y : enforce orthogonal constraint on Y (one-side orthogonal) 
        AY : enforce orthogonal constraint on A and Y (bi orthogonal)
        'AY' (default) 

    Attributes
    ----------
    A : "data_dimension x num_bases" matrix of basis vectors
    S : "num_bases x num_bases" matrix, absorb the values due to orthonormality of A and Y.
    Y : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())  

    """

    def __init__(self, data, num_bases=4, niter=10, orthogonal='AY', **kwargs):
        PyNMFBase.__init__(self, data, num_bases, niter, **kwargs)
        self.orthogonal = orthogonal

    def _init_s(self):
        if (self.orthogonal=='AY'):
            self.S= np.dot(np.dot(self.A.T, self.data[:,:]), self.Y.T)
        else:
            self.S= np.eye(self._num_bases, self._num_bases)

    def _update_a(self):
        # pre init A1, and A2 (necessary for storing matrices on disk)
        if (self.orthogonal == 'A' or self.orthogonal == 'AY'):
            A1 = np.dot(np.dot(np.dot(np.dot(self.A, self.A.T), self.data[:,:]), self.Y.T), self.S.T) + 10**-9
            A2 = np.sqrt(np.dot(np.dot(self.data[:,:], self.Y.T), self.S.T)/A1)
            self.A *= A2
            
        elif (self.orthogonal == 'Y'):
            A1 = np.dot(np.dot(self.A, self.Y), self.Y.T) + 10**-9
            A2 = np.dot(self.data[:,:], self.Y.T)/A1
            self.A *= A2
            
            # to normalize
            # self.A /= np.sqrt(np.sum(self.A**2.0, axis=0))

        else:
            print("please insert orthogonal constraint")

    def _update_y(self):
        # pre init Y1, and Y2 (necessary for storing matrices on disk)
        if (self.orthogonal == 'Y' or self.orthogonal == 'AY'):
            Y1 = np.dot(np.dot(np.dot(self.S.T, self.A.T), self.data[:,:]), np.dot(self.Y.T, self.Y))+ 10**-9
            Y2 = np.sqrt(np.dot(np.dot(self.S.T, self.A.T), self.data[:,:])/Y1)
            self.Y *= Y2

        elif (self.orthogonal == 'A'):
            Y1 = np.dot(np.dot(self.A.T, self.A), self.Y) + 10**-9
            Y2 = np.dot(self.A.T, self.data[:,:])/Y1
            self.Y *= Y2
            # to normalize
            # self.Y /= np.sqrt(np.sum(self.Y**2.0, axis=0))
            
        else:
            print("please insert orthogonal constraint")

    def _update_s(self):
        if (self.orthogonal == 'AY'):
            S1 = np.dot(np.dot(np.dot(self.A.T, self.A), self.S), np.dot(self.Y, self.Y.T)) + 10**-9
            S2 = np.sqrt(np.dot(np.dot(self.A.T, self.data[:,:]), self.Y.T)/S1)
            self.S *= S2
        else:
            self.S = self.S


    def frobenius_norm(self):
        """ Frobenius norm (||data - ASY||) of a data matrix and a low rank
        approximation given by ASY. Minimizing the Fnorm ist the most common
        optimization criterion for matrix factorization methods.

        Returns:
        -------
        frobenius norm: F = || data - A*S*Y||

        """
        # check if A S and Y exist
        if hasattr(self,'A') and hasattr(self,'S') and hasattr(self,'Y'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:,:] - ((self.A * self.S) * self.Y)
                tmp = tmp.multiply(tmp).sum()
                #err = tmp
                err = np.sqrt(tmp)
            else:
                #err = np.sum((self.data[:,:] - np.dot(np.dot(self.A, self.S), self.Y))**2 )
                err = np.sqrt( np.sum((self.data[:,:] - np.dot(np.dot(self.A, self.S), self.Y))**2 ))
        else:
            err = None

        return err            
    
    def factorize(self, niter=100, show_progress=False, compute_a=True, compute_y=True, compute_s=True, compute_err=True, epoch_hook=None):
        """ Factorize s.t. ASY = data

        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_a : bool
                iteratively update values for A.
        compute_y : bool
                iteratively update values for Y.
        compute_s : bool
                iteratively update values for S.
        compute_err : bool
                compute Frobenius norm |data-ASY| after each update and store
                it to .ferr[k].
        epoch_hook : function
                If this exists, evaluate it every iteration

        Updated Values
        --------------
        .A : updated values for A.
        .Y : updated values for Y.
        .S : updated values for S.
        .ferr : Frobenius norm |data-ASY| for each iteration.
        """

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        # create W and H if they don't already exist
        # -> any custom initialization to W,H and S should be done before
        if not hasattr(self,'A') and compute_a:
            self._init_a()

        if not hasattr(self,'Y') and compute_y:
            self._init_y()

        if not hasattr(self,'S') and compute_s:
            self._init_s()

        # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(self._niter)

        for i in range(self._niter):
            if compute_a:
                self._update_a()

            if compute_y:
                self._update_y()

            if compute_s:
                self._update_s()

            if compute_err:
                # compute the error using self.frobenius_norm() for frobenius_norm
                self.ferr[i] = self.frobenius_norm()
                self._logger.info('FN: %s (%s/%s)'  %(self.ferr[i], i+1, self._niter))
                # print(('FN: %s (%s/%s)'  %(self.ferr[i], i+1, self._niter)))
            else:
                self._logger.info('Iteration: (%s/%s)'  %(i+1, self._niter))

            if epoch_hook is not None:
                epoch_hook(self)


            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self._converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break


def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
