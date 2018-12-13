from os.path import dirname, abspath, join
from warnings import warn
import numpy as np
np.set_printoptions(threshold=np.nan)
from pynmf.nmf import NMF
from pynmf.vsnmf import VSNMF
from pynmf.orthogonal import ORTHOGONAL
from pynmf.biorthogonal import BIORTHOGONAL
from pynmf.orthogonal_ridge import L2ORTHOGONAL

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import time

"""
    .. table:: Standard Cluster assignments obtained with this example for rank = 2 and rank = 3. 

       ====================  ========== ==========
              Sample          rank = 2   rank = 3
       ====================  ========== ==========
        ALL_19769_B-cell        0            0
        ALL_23953_B-cell        0            0
        ALL_28373_B-cell        0            0
        ALL_9335_B-cell         0            0
        ALL_9692_B-cell         0            0
        ALL_14749_B-cell        0            0
        ALL_17281_B-cell        0            0
        ALL_19183_B-cell        0            0
        ALL_20414_B-cell        0            0
        ALL_21302_B-cell        0            0
        ALL_549_B-cell          0            0
        ALL_17929_B-cell        0            0
        ALL_20185_B-cell        0            0
        ALL_11103_B-cell        0            0
        ALL_18239_B-cell        0            0
        ALL_5982_B-cell         0            0
        ALL_7092_B-cell         0            0
        ALL_R11_B-cell          0            0
        ALL_R23_B-cell          0            0
        ALL_16415_T-cell        0            1
        ALL_19881_T-cell        0            1
        ALL_9186_T-cell         0            1
        ALL_9723_T-cell         0            1
        ALL_17269_T-cell        0            1
        ALL_14402_T-cell        0            1
        ALL_17638_T-cell        0            1
        ALL_22474_T-cell        0            1       
        AML_12                  1            2
        AML_13                  1            2
        AML_14                  1            2
        AML_16                  1            2
        AML_20                  1            2
        AML_1                   1            2
        AML_2                   1            2
        AML_3                   1            2
        AML_5                   1            2 
        AML_6                   1            2
        AML_7                   1            2
       ====================  ========== ========== 
"""

def read():
    """
    Read ALL AML gene expression data. The matrix's shape is 5000 (genes) x 38 (samples).
    It contains only positive data.

    Return the gene expression data matrix.
    """
    fname = join(dirname(abspath(__file__)), 'datasets', 'ALL_AML', 'ALL_AML_data.txt')
    V = np.loadtxt(fname)
    return V

def run():
    """Run Standard NMF on leukemia data set. """
    time_start = time.perf_counter()
    data = read()
    #print("V shape", V.shape)
    
    #Basic NMF
    #nmf_mdl = NMF(data, num_bases=2, niter=100)
    
    #VSNMF
    #nmf_mdl = VSNMF(data, num_bases=2, niter=100, alfa1=0, alfa2=0, lambda1=0, lambda2=0, t1=1, t2=1)

    #Orthogonal NMF
    #nmf_mdl = ORTHOGONAL(data, num_bases=2, niter=100, orthogonal='Y')
    
    #Orthogonal NonNegative Tri-Factorization
    #nmf_mdl = BIORTHOGONAL(data, num_bases=2, niter=100, orthogonal='AY')

    #L2Orthogonal NMF
    #nmf_mdl=L2ORTHOGONAL(data, num_bases=2, niter=100, orthogonal='Y', alpha_a=0.1, alpha_y=0.1)
    
    nmf_mdl.factorize()
    # print(data)
    # print(nmf_mdl.A)
    # print(nmf_mdl.Y)
    cluster_result = np.argmax(nmf_mdl.Y, 0)
    print('SSE', nmf_mdl.ferr[-1])
    print('Cluster', cluster_result)

    time_elapsed = (time.perf_counter() - time_start)
    print('Average Time', time_elapsed)
     

if __name__ == "__main__":
    """Run the ALL AML example."""
    run()
