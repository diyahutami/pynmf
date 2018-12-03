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
    Orthogonal Non-negative Matrix Tri-factorizations for Clustering
    Orthogonal NMF factorize a data matrix into 3 matrices
    s.t. F = | data - W*S*H | = | is minimal. W, S and H are restricted to non-negativ data. 
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
    W : "data_dimension x num_bases" matrix of basis vectors
    S: "num_bases x num_bases" matrix, absorb the values due to orthonormality of W and H.
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize())  

    """

    def __init__(self, data, num_bases=4, niter=10, orthogonal='AY', **kwargs):
        PyNMFBase.__init__(self, data, num_bases, niter, **kwargs)
        self.orthogonal = orthogonal

    def _init_s(self):
        if (self.orthogonal=='AY'):
            self.S= np.dot(np.dot(self.W.T, self.data[:,:]), self.H.T)
        else:
            self.S= np.eye(self._num_bases, self._num_bases)

    def _update_w(self):
        # pre init W1, and W2 (necessary for storing matrices on disk)
        if (self.orthogonal == 'A' or self.orthogonal == 'AY'):
            W1 = np.dot(np.dot(np.dot(np.dot(self.W, self.W.T), self.data[:,:]), self.H.T), self.S.T) + 10**-9
            W2= np.dot(np.dot(self.data[:,:], self.H.T), self.S.T)/W1
            self.W *= W2
            
        elif (self.orthogonal == 'Y'):
            W1 = np.dot(np.dot(self.W, self.H), self.H.T) + 10**-9
            W2 = np.dot(self.data[:,:], self.H.T)/W1
            self.W *= W2
            
            # to normalize
            # self.W /= np.sqrt(np.sum(self.W**2.0, axis=0))

        else:
            print("please insert orthogonal constraint")

    def _update_h(self):
        # pre init H1, and H2 (necessary for storing matrices on disk)
        if (self.orthogonal == 'Y' or self.orthogonal == 'AY'):
            H1 = np.dot(np.dot(np.dot(self.S.T, self.W.T), self.data[:,:]), np.dot(self.H.T, self.H))+ 10**-9
            H2 = np.dot(np.dot(self.S.T, self.W.T), self.data[:,:])/H1
            self.H *= H2

        elif (self.orthogonal == 'A'):
            H1 = np.dot(np.dot(self.W.T, self.W), self.H) + 10**-9
            H2 = np.dot(self.W.T, self.data[:,:])/H1
            self.H *= H2
            # to normalize
            # self.H /= np.sqrt(np.sum(self.H**2.0, axis=0))
            
        else:
            print("please insert orthogonal constraint")

    def _update_s(self):
        if (self.orthogonal == 'AY'):
            S1 = np.dot(np.dot(np.dot(self.W.T, self.W), self.S), np.dot(self.H, self.H.T)) + 10**-9
            S2 = np.dot(np.dot(self.W.T, self.data[:,:]), self.H.T)/S1
            self.S *= S2
        else:
            self.S = self.S


    def frobenius_norm(self):
        """ Frobenius norm (||data - WSH||) of a data matrix and a low rank
        approximation given by WSH. Minimizing the Fnorm ist the most common
        optimization criterion for matrix factorization methods.

        Returns:
        -------
        frobenius norm: F = || data - W*S*H||

        """
        # check if W S and H exist
        if hasattr(self,'W') and hasattr(self,'S') and hasattr(self,'H'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:,:] - ((self.W * self.S) * self.H)
                tmp = tmp.multiply(tmp).sum()
                err = tmp
                #err = np.sqrt(tmp)
            else:
                err = np.sum((self.data[:,:] - np.dot(np.dot(self.W, self.S), self.H))**2 )
                #err = np.sqrt( np.sum((self.data[:,:] - np.dot(np.dot(self.W, self.S), self.H))**2 ))
        else:
            err = None

        return err            
    
    def factorize(self, niter=100, show_progress=False, compute_w=True, compute_h=True, compute_s=True, compute_err=True, epoch_hook=None):
        """ Factorize s.t. WsH = data

        Parameters
        ----------
        niter : int
                number of iterations.
        show_progress : bool
                print some extra information to stdout.
        compute_w : bool
                iteratively update values for W.
        compute_h : bool
                iteratively update values for H.
        compute_s : bool
                iteratively update values for S.
        compute_err : bool
                compute Frobenius norm |data-WSH| after each update and store
                it to .ferr[k].
        epoch_hook : function
                If this exists, evaluate it every iteration

        Updated Values
        --------------
        .W : updated values for W.
        .H : updated values for H.
        .S : updated values for S.
        .ferr : Frobenius norm |data-WSH| for each iteration.
        """

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        # create W and H if they don't already exist
        # -> any custom initialization to W,H and S should be done before
        if not hasattr(self,'W') and compute_w:
            self._init_w()

        if not hasattr(self,'H') and compute_h:
            self._init_h()

        if not hasattr(self,'S') and compute_s:
            self._init_s()

        # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(self._niter)

        for i in range(self._niter):
            if compute_w:
                self._update_w()

            if compute_h:
                self._update_h()

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
