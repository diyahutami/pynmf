from os.path import dirname, abspath, join
from warnings import warn
import numpy as np
np.set_printoptions(threshold=np.nan)
from pynmf.nmf import NMF
from pynmf.vsnmf import VSNMF
from pynmf.orthogonal import ORTHOGONAL

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import time

"""
   .. table:: Standard NMF Class assignments results obtained with this example for rank = 2, rank = 3 and rank = 5.  

       ====================  ========== ========== ========== ==========
              Sample           Class     rank = 2   rank = 3   rank = 5 
       ====================  ========== ========== ========== ==========
        Brain_MD_7                C        0            1        3
        Brain_MD_59               C        1            0        2
        Brain_MD_20               C        1            1        3
        Brain_MD_21               C        1            1        3
        Brain_MD_50               C        1            1        4
        Brain_MD_49               C        0            2        3
        Brain_MD_45               C        1            1        3
        Brain_MD_43               C        1            1        3
        Brain_MD_8                C        1            1        3
        Brain_MD_42               C        0            2        4
        Brain_MD_1                C        0            2        3
        Brain_MD_4                C        0            2        3 
        Brain_MD_55               C        0            2        3
        Brain_MD_41               C        1            1        2
        Brain_MD_37               C        1            0        3
        Brain_MD_3                C        1            2        3
        Brain_MD_34               C        1            2        4
        Brain_MD_29               C        1            1        2
        Brain_MD_13               C        0            1        2
        Brain_MD_24               C        0            1        3
        Brain_MD_65               C        1            0        2
        Brain_MD_5                C        1            0        1
        Brain_MD_66               C        1            0        1
        Brain_MD_67               C        1            0        3
        Brain_MD_58               C        0            2        3
        Brain_MD_53               D        0            2        4
        Brain_MD_56               D        0            2        4
        Brain_MD_16               D        0            2        4
        Brain_MD_40               D        0            1        0
        Brain_MD_35               D        0            2        4
        Brain_MD_30               D        0            2        4
        Brain_MD_23               D        0            2        4
        Brain_MD_28               D        1            2        1
        Brain_MD_60               D        1            0        0
       ====================  ========== ========== ========== ==========  
"""

def read(normalize=False):
    """
    Read the medulloblastoma gene expression data. The matrix's shape is 5893 (genes) x 34 (samples). 
    It contains only positive data.
    
    Return the gene expression data matrix. 
    """
    fname = join(dirname(abspath(__file__)), 'datasets', 'Medulloblastoma',  'Medulloblastoma_data.txt')
    V = np.loadtxt(fname)
    if normalize:
        V = (V - V.min()) / (V.max() - V.min())
    return V


def run():
    time_start = time.perf_counter()
    data = read()
    #print("V shape", V.shape)
    
    #Basic NMF
    nmf_mdl = NMF(data, num_bases=2, niter=100)
    
    #VSNMF
    #nmf_mdl = VSNMF(data, num_bases=2, niter=100, alfa1=0, alfa2=0, lambda1=0, lambda2=0, t1=1, t2=1)
    
    #Orthogonal NMF
    #nmf_mdl = ORTHOGONAL(data, num_bases=2, niter=100, orthogonal='A')
    
    nmf_mdl.factorize()
    # print(data)
    # print(nmf_mdl.W)
    # print(nmf_mdl.H)
    cluster_result = np.argmax(nmf_mdl.H, 0)
    print('SSE', nmf_mdl.ferr[-1])
    print('Cluster', cluster_result)

    time_elapsed = (time.perf_counter() - time_start)
    print('Average Time', time_elapsed)

if __name__ == "__main__":
    """Run the medulloblastoma example."""
    run()
