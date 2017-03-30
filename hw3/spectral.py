import numpy as np
from numpy import linalg as LA

def main():
    """
    Run spectral clustering on the data in the problem.
    """
    m = np.array([[0, 4, 3, 0, 0],
                  [4, 0, 2, 0, 0],
                  [3, 2, 0, 1, 2],
                  [0, 0, 1, 0, 0],
                  [0, 0, 2, 0, 0]], dtype=np.float64)

    # copy the array
    l = np.array(m)

    # Create the laplacian matrix
    for i in range(len(l)):
        l[i, i] = -sum(l[i])
        l[i] = -l[i]

    eig_val, eig_vec = LA.eig(l)
    # For some reason column i corresponds to the ith eigenvector
    # Take the transverse to simplify access from eig_vec[:,i]
    # to eig_vec[i]
    eig_vec = eig_vec.T

    # sort the indexes based on the eigenvalue of the index
    idx_sort = sorted(range(len(eig_val)), key=lambda idx: eig_val[idx])

    clust_conns = eig_vec[idx_sort[1]]

    med = np.median(clust_conns)

    print("Matix:\n", m)
    print("Laplacian Matrix:\n", l)
    print("Eigenvalues:\n",
          eig_val)
    print("Eigenvectors:\n",
          eig_vec)
    print("Ordered eigenvalue indexes:\n",
          idx_sort)
    print("Ordered eigenvalues:\n",
          eig_val[idx_sort])
    print("Cluster values (vector of second smallest eigenvalue):\n",
          clust_conns)
    print("Cluster allocations:\n",
          np.sign(clust_conns))
    print("Ordered cluster values:\n",
          sorted(range(len(clust_conns)), key=lambda idx: clust_conns[idx]))
    print("Cluster values around median:\n",
          clust_conns - med)
    print("Cluster allocations around median:\n",
          np.sign(clust_conns - med))

if __name__ == "__main__":
    main()
