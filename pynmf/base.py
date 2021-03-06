"""
PyNMF base class used in (almost) all matrix factorization methods

"""
import numpy as np
import logging
import logging.config
import scipy.sparse
from numpy.linalg import eigh
from scipy.misc import factorial
from .dist import *

__all__ = ["PyNMFBase", "PyNMFBase3", "eighk", "cmdet", "simplex"]
_EPS = np.finfo(float).eps

def eighk(M, k=0):
    """ Returns ordered eigenvectors of a squared matrix. Too low eigenvectors
    are ignored. Optionally only the first k vectors/values are returned.

    Arguments
    ---------
    M - squared matrix
    k - (default 0): number of eigenvectors/values to return

    Returns
    -------
    w : [:k] eigenvalues
    v : [:k] eigenvectors

    """
    values, vectors = eigh(M)

    # get rid of too low eigenvalues
    s = np.AYere(values > _EPS)[0]
    vectors = vectors[:, s]
    values = values[s]

    # sort eigenvectors according to largest value
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:,idx]

    # select only the top k eigenvectors
    if k > 0:
        values = values[:k]
        vectors = vectors[:,:k]

    return values, vectors


def cmdet(d):
    """ Returns the Volume of a simplex computed via the Cayley-Menger
    determinant.

    Arguments
    ---------
    d - euclidean distance matrix (shouldn't be squared)

    Returns
    -------
    V - volume of the simplex given by d
    """
    D = np.ones((d.shape[0]+1,d.shape[0]+1))
    D[0,0] = 0.0
    D[1:,1:] = d**2
    j = np.float32(D.shape[0]-2)
    f1 = (-1.0)**(j+1) / ( (2**j) * ((factorial(j))**2))
    cmd = f1 * np.linalg.det(D)

    # sometimes, for very small values, "cmd" might be negative, thus we take
    # the absolute value
    return np.sqrt(np.abs(cmd))


def simplex(d):
    """ Computed the volume of a simplex S given by a coordinate matrix D.

    Arguments
    ---------
    d - coordinate matrix (k x n, n samples in k dimensions)

    Returns
    -------
    V - volume of the Simplex spanned by d
    """
    # compute the simplex volume using coordinates
    D = np.ones((d.shape[0]+1, d.shape[1]))
    D[1:,:] = d
    V = np.abs(np.linalg.det(D)) / factorial(d.shape[1] - 1)
    return V


class PyNMFBase():
    """
    PyNMF Base Class. Does nothing useful apart from providing some basic methods.
    """
    # some small value

    _EPS = _EPS

    def __init__(self, data, num_bases=4, niter=10, **kwargs):
        """
        """

        def setup_logging():
            # create logger
            self._logger = logging.getLogger("pymf")

            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()

        # set variables
        self.data = data
        self._num_bases = num_bases
        self._niter = niter

        # initialize H and W to random values
        self._data_dimension, self._num_samples = self.data.shape


    def residual(self):
        """ Returns the residual in % of the total amount of data

        Returns
        -------
        residual : float
        """
        res = np.sum(np.abs(self.data - np.dot(self.A, self.Y)))
        total = 100.0*res/np.sum(np.abs(self.data))
        return total

    def frobenius_norm(self):
        """ Frobenius norm (||data - AY||) of a data matrix and a low rank
        approximation given by AY. Minimizing the Fnorm ist the most common
        optimization criterion for matrix factorization methods.

        Returns:
        -------
        frobenius norm: F = ||data - AY||

        """
        # check if A and Y exist
        if hasattr(self,'A') and hasattr(self,'Y'):
            if scipy.sparse.issparse(self.data):
                tmp = self.data[:,:] - (self.A * self.Y)
                tmp = tmp.multiply(tmp).sum()
                #err = tmp
                err = np.sqrt(tmp)
            else:
                #err = np.sum((self.data[:,:] - np.dot(self.A, self.Y))**2 )
                err = np.sqrt( np.sum((self.data[:,:] - np.dot(self.A, self.Y))**2 ))
        else:
            err = None

        return err

    def divergence(self):
        """ Kullback-Leiber divergence between two matrices
        Returns:
        -------
        divergence: F = D(X||AY)

        """
        # check if A and Y exist
        if hasattr(self,'A') and hasattr(self,'Y'):
            if scipy.sparse.issparse(self.data):
                d = self.A * self.Y
                tmp = kl_divergence(d, self.data[:,:])
                tmp = tmp.sum()
                err = tmp
                #err = np.sqrt(tmp)
            else:
                err = np.sum(kl_divergence(np.dot(self.A, self.Y), self.data[:,:]))
                #err = np.sqrt(np.sum(kl_divergence(np.dot(self.A, self.Y), self.data[:,:])))
        else:
            err = None

        return err

    def _init_a(self):
        """ Initalize A to random values [0,1].
        """
        # add a small value, otherwise nmf and related methods get into trouble as
        # they have difficulties recovering from zero.
        self.A = np.random.random((self._data_dimension, self._num_bases)) + 10**-4

    def _init_y(self):
        """ Initalize Y to random values [0,1].
        """
        self.Y = np.random.random((self._num_bases, self._num_samples)) + 10**-4

    def _update_y(self):
        """ Overwrite for updating Y.
        """
        pass

    def _update_a(self):
        """ Overwrite for updating A.
        """
        pass

    def _converged(self, i):
        """
        If the optimization of the approximation is below the machine precision,
        return True.

        Parameters
        ----------
            i   : index of the update step

        Returns
        -------
            converged : boolean
        """
        derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100, show_progress=False, compute_a=True, compute_y=True, compute_err=True, epoch_hook=None):
        """ Factorize s.t. AY = data

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
        compute_err : bool
                compute Frobenius norm |data-AY| after each update and store
                it to .ferr[k].
        epoch_hook : function
                If this exists, evaluate it every iteration

        Updated Values
        --------------
        .A : updated values for A.
        .Y : updated values for Y.
        .ferr : Frobenius norm |data-AY| for each iteration.
        """

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        # create W and H if they don't already exist
        # -> any custom initialization to A,Y should be done before
        if not hasattr(self,'A') and compute_a:
            self._init_a()

        if not hasattr(self,'Y') and compute_y:
            self._init_y()

        # Computation of the error can take quite long for large matrices,
        # thus we make it optional.
        if compute_err:
            self.ferr = np.zeros(self._niter)

        for i in range(self._niter):
            if compute_a:
                self._update_a()

            if compute_y:
                self._update_y()

            if compute_err:
                # compute the error using self.frobenius_norm() for frobenius_norm
                # compute the error using self.divergence for Kullback-Leiber divergence
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


class PyNMFBase3():
    """
    PyNMFBase3(data, show_progress=False)

    Base class for factorizing a data matrix into three matrices s.t.
    F = | data - USV| is minimal (e.g. SVD, CUR, ..)

    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data

    Attributes
    ----------
        U,S,V : submatrices s.t. data = USV

    """
    _EPS = _EPS


    def __init__(self, data, k=-1, rrank=0, crank=0):
        """
        """
        self.data = data
        (self._rows, self._cols) = self.data.shape

        self._rrank = self._rows
        if rrank > 0:
            self._rrank = rrank

        self._crank = self._cols
        if crank > 0:
            self._crank = crank

        self._k = k

    def frobenius_norm(self):
        """ Frobenius norm (||data - USV||) for a data matrix and a low rank
        approximation given by SVH using rank k for U and V

        Returns:
            frobenius norm: F = ||data - USV||
        """
        if scipy.sparse.issparse(self.data):
            err = self.data - (self.U*self.S*self.V)
            err = err.multiply(err)
            err = np.sqrt(err.sum())
        else:
            err = self.data[:,:] - np.dot(np.dot(self.U, self.S), self.V)
            err = np.sqrt(np.sum(err**2))

        return err


    def factorize(self):
        pass

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
