import numpy as np

from scipy.linalg import det
from sklearn.decomposition import PCA, TruncatedSVD

import warnings

class NFINDR:

    """
    N-FINDR algorithm implementation
        
    Ref: Winter, M. E. (1999). N-FINDR: An algorithm for fast autonomous 
    spectral endmember determination in hyperspectral data. In: Imaging 
    Spectrometry V (Vol. 3753, pp. 266-275). International Society for Optics 
    and Photonics.
    
    Parameters
    ----------
        
    Y: numpy array (size: n_bands by n_samples)
      data matrix
    n_sources: int
      number of endmembers
          
    Attributes
    ----------

    Y: numpy array (size: n_bands by n_samples)
      data points
    n_samples: int
      number of samples in the dataset
    n_sources: int
      number of endmembers
    n_bands: int
      spectral dimension
    Y_proj: numpy array (size: n_sample by n_sources-1 matrix)
      data points in the reduced dimension space 
    M_proj: numpy array (size: n_sources by n_sources-1 matrix)
      endmembers vectors in the reduced dimension space 
    vol: float
      volume of the simplex formed by the endmembers
    M: numpy array (size: n_bands by n_sources)
      endmembers matrix 
    """

    def __init__(self, Y, n_sources):

        warnings.warn("Y must be of shape (n_bands, n_samples). Please verify your input.")

        self.Y = Y
        self.n_bands, self.n_samples = self.Y.shape[0], self.Y.shape[1]
        self.n_sources = n_sources

        # Project the data points
        pca = PCA(n_components=self.n_sources - 1)
        self.Y_proj = pca.fit_transform(Y.T)
        
        # Select random data points as initial guess for the endmembers
        random_indices = np.random.choice(self.n_samples, size=self.n_sources, 
          replace=False)
        self.M_proj = self.Y_proj[random_indices, :]

        # Computes the volume of the endmembers simplex
        self.vol = abs(det(self.M_proj[1:, :] - self.M_proj[0, :]))


    def run(self):

        """
        Run the N-FINDR algorithm
        """
        indexes = []
        for s in range(self.n_sources):

            endmembers = np.copy(self.M_proj)
            idx = 0

            # Iterate over the data points
            for n in range(self.n_samples):

                # Try replacing the selected endmember by the data point
                endmembers[s, :] = self.Y_proj[n, :]
                vol = abs(det(endmembers[1:, :] - endmembers[0, :]))

                # Update the endmember if the volume is greater than 
                # the current one
                if(vol > self.vol):
                    self.M_proj = np.copy(endmembers)
                    self.vol = vol
                    idx = n

            indexes.append(idx)

        self.M = self.Y[:, np.array(indexes)]
        
        
class VCA:

    """
    Vertex Component Analysis implementation
        
    Ref: Winter, M. E. (1999). N-FINDR: An algorithm for fast autonomous 
    spectral endmember determination in hyperspectral data. In: Imaging 
    Spectrometry V (Vol. 3753, pp. 266-275). International Society for Optics 
    and Photonics.
    
    Parameters
    ----------
        
    Y: numpy array (size: n_bands by n_samples)
      data matrix
    n_sources: int
      number of endmembers
    method: string
      method used to perform the projection on the endmembers subspace
          
    Attributes
    ----------

    Y: numpy array (size: n_bands by n_samples)
      data points
    n_samples: int
      number of samples in the dataset
    n_sources: int
      number of endmembers
    n_bands: int
      spectral dimension
    method: string
      method used to identify the subspace
    Y_proj: numpy array (size: n_sample by n_sources-1 matrix)
      data points in the reduced dimension space 
    M_proj: numpy array (size: n_sources by n_sources-1 matrix)
      endmembers vectors in the reduced dimension space 
    vol: float
      volume of the simplex formed by the endmembers
    M: numpy array (size: n_bands by n_sources)
      endmembers matrix 
    """

    def __init__(self, Y, n_sources, method='pca'):

        warnings.warn("Y must be of shape (n_bands, n_samples). Please verify your input.")

        self.Y = Y
        self.n_bands, self.n_samples = self.Y.shape
        self.n_sources = n_sources
        self.method = method
        
        # Data projection
        self.Y_proj = self.project(Y, n_sources, method)
    
    def project(self, Y, n_sources, method='pca'):

        """
        Project data on the specified subspace
        """
        if method == "pca":
            pca = PCA(n_components = n_sources)
            Y_proj = pca.fit_transform(Y.T)
        elif method == "svd":
            svd = TruncatedSVD(n_components = n_sources)
            Y_proj = svd.fit_transform(Y.T)
        else:
            raise ValueError("Invalid method. Choose 'pca' or 'ica'")

        return Y_proj

    def run(self):

        """
        Run VCA algorithm
        """
        
        indice = np.zeros((self.n_sources), dtype=int)
        A = np.zeros((self.n_sources, self.n_sources))
        A[-1, 0] = 1

        for i in range(self.n_sources):
        
            w = np.random.rand(self.n_sources, 1)
            f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
            f = f / np.linalg.norm(f)
            v = np.dot(f.T, self.Y_proj.T)

            indice[i] = np.argmax(np.abs(v))
            A[:, i] = self.Y_proj.T[:, indice[i]]

        Ae = self.Y[:, indice]
        return Ae


if __name__ == '__main__':

    # Test N-FINDR
    Y = np.random.rand(10, 50)
    n_sources = 3
    nfdr = NFINDR(Y, n_sources)
    nfdr.run()

    # Test VCA
    vca = VCA(Y, n_sources)
    vca.run()