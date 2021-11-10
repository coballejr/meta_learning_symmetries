import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
    
class Filter:
    
    def __init__(self, size, values):
        assert size % 2 == 1, 'Filter size must be odd.'
        assert size == len(values), 'Length of values must match size.'
        self.size = size
        self.h = int((size - 1) / 2)
        self.values = values
        
    def convolve(self, f):
        #breakpoint()
        signal_size = len(f)
        h = np.zeros(signal_size)
        for i in range(signal_size):
            first_ind, last_ind = (i - self.h), (i + self.h)
            inds = np.linspace(first_ind, last_ind, self.size, dtype = int) % signal_size
            f_window = f[inds]
            val = (self.values*f_window).sum()
            h[i] = val
        return h
    
    def show_convolution(self, f):
        h = self.convolve(f)
        fig, ax = plt.subplots(1,3)
        ax[0].plot(f)
        ax[0].set_title('Signal')
        ax[1].plot(self.values)
        ax[1].set_title('Filter')
        ax[2].plot(h)
        ax[2].set_title('Convolution')
        plt.tight_layout()
        plt.show()
        plt.close()
        return h
    
class LocalPCA:
    
    def __init__(self, dim, k):
        self.pca = PCA(n_components = dim, whiten = True)
        self.knn = NearestNeighbors(n_neighbors = k)
        
    def fit(self, X):
        frames = []
        self.knn.fit(X)
        groups = self.knn.kneighbors_graph(X).toarray()
        for group in groups:
            group = [i for i in range(len(group)) if group[i] == 1]
            group = np.array(group, dtype = int)
            x = X[group]
            self.pca.fit(x)
            vecs = self.pca.components_
            frames.append(vecs)
        return frames
    
    def show_frame(self, inds, X):
        frames = self.fit(X)
        plt.scatter(X[:,0], X[:,1], s = 2)
        for i in inds:
            x, y = X[i]
            dx, dy = frames[i].flatten()
            plt.arrow(x, y, dx, dy)
        plt.gca().set_aspect('equal')
        plt.show()
        plt.close()
        return frames
    

def _normal2d(vecs):
    return np.vstack([-vecs[:,1], vecs[:,0]]).T

def _normal3d(vecs):
    A, B = np.array([v[0] for v in vecs]), np.array([v[1] for v in vecs])
    return np.cross(A, B)
    
def _projSxM(x, normal):
    
    projx = np.outer(x,x)
    projNx = np.outer(normal, normal)
    
    dim = x.shape[0]
    proj = np.zeros((dim**2,dim**2))
    std_basis = np.eye(dim)
    
    basis_ind = 0
    for i in range(dim):
        for j in range(dim):
            basis_elem = np.outer(std_basis[i,:], std_basis[j,:])
            img_of_basis_elem = projNx @ basis_elem @ projx
            img_of_basis_elem = img_of_basis_elem.flatten()
            proj[:,basis_ind] = img_of_basis_elem
            basis_ind += 1
    return proj

def _sum_projSxM(X, normals):
    dim = X.shape[1]
    sum_proj = np.zeros((dim**2,dim**2))
    for x, normal in zip(X, normals):
        sum_proj += _projSxM(x, normal)
    return sum_proj