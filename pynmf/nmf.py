"""
PyNMF Non-negative Matrix Factorization.

    NMF: Class for Non-negative Matrix Factorization

[1] D. D. Lee and H. S. Seung, "Learning the Parts of Objects by Non-negative Matrix Factorization",
 Nature, vol.401, no. 6755, pp. 788-791, 1999
[2] J.-P. Brunet, P. Tamayo, T. R. Golub, and J.P. Mesirov, "Metagenes and molecular pattern discovery 
using matrix factorization”, Proc. Nat. Academy Sci. United States Ameerica, vol.101, no.12, PP. 4164-4169, 2004

"""
import numpy as np
import logging
import logging.config
import scipy.sparse
from .base import PyNMFBase

__all__ = ["NMF"]


class NMF(PyNMFBase):
    """
    NMF(data, num_bases=4, niter=10)

    Non-negative Matrix Factorization. Factorize a data matrix into two matrices
    s.t. F = | data - A*Y | = | is minimal. A, and Y are restricted to non-negative
    data. Uses the classicial multiplicative update rule.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of A and row rank of Y).
        4 (default) 
    niter : int, optional
        Number of iteration
        10 (default)       

    Attributes
    ----------
    A : "data_dimension x num_bases" matrix of basis vectors
    Y : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize()) 

    Example
    -------
    Applying NMF to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> nmf_mdl = NMF(data, num_bases=2, niter=10)
    >>> nmf_mdl.factorize()

    The basis vectors are now stored in nmf_mdl.A, the coefficients in nmf_mdl.Y.
    To compute coefficients for an existing set of basis vectors simply copy A
    to nmf_mdl.A, and set compute_a to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMF(data, num_bases=2)
    >>> nmf_mdl.A = A
    >>> nmf_mdl.factorize(niter=20, compute_a=False)

    The result is a set of coefficients nmf_mdl.Y, s.t. data = A * nmf_mdl.Y.
    """
    def _update_a(self):
        # pre init A1 (necessary for storing matrices on disk)
        A1 = np.dot(np.dot(self.A, self.Y), self.Y.T) + 10**-9
        self.A *= np.dot(self.data[:,:], self.Y.T)
        self.A /= A1
        # to normalize
        # self.A /= np.sqrt(np.sum(self.A**2.0, axis=0))

    def _update_y(self):
        # pre init Y1 (necessary for storing matrices on disk)
        Y1 = np.dot(np.dot(self.A.T, self.A), self.Y) + 10**-9
        self.Y *= np.dot(self.A.T, self.data[:,:])
        self.Y /= Y1
        # to normalize
        # self.Y /= np.sqrt(np.sum(self.Y**2.0, axis=0))

def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
