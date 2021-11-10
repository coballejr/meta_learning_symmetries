import numpy as np
from scipy.special import factorial2
import matplotlib.pyplot as plt

class Sphere:
    
    def __init__(self, dim, r):
        self.dim = dim
        self.r = r
        if dim % 2 == 1:
            self.area = 2**((self.dim + 1)/2)*np.pi**((self.dim - 1)/2) / (factorial2(dim - 2))
        else:
            self.area = (2*(np.pi)**(self.dim / 2))/ np.math.factorial(int(0.5*self.dim - 1))
            
    def sample(self, nsamps, noise = 0):
        u = 2*np.random.rand(nsamps, self.dim)-1
        unorm = np.linalg.norm(u, keepdims = True, axis = 1)
        s = u / unorm
        if noise:
            w = np.random.randn(nsamps, self.dim)
            s += noise*w
        return s
    
    def grid(self, n):
        assert self.dim == 2, 'Grid only supported for dim = 2.'
        endpoint = (2*np.pi*(n - 1))/ n
        theta = np.linspace(0, endpoint, n)
        x, y = self.r*np.cos(theta), self.r*np.sin(theta)
        circle = np.vstack([x,y]).T
        return theta, circle
    
    def _show_circle(self,circle):
        x, y = circle[:,0], circle[:,1]
        plt.scatter(x, y)            
        plt.gca().set_aspect('equal')
        plt.show()
        plt.close()
    
    def show_grid(self, n):
        assert self.dim == 2, 'Plotting only supported for dim = 2.'
        theta, circle = self.grid(n)
        self._show_circle(circle)
        return theta, circle 
        
    def show_samples(self, nsamps, noise = 0):
        assert self.dim == 2, 'Plotting only supported for dim = 2.'
        s = self.sample(nsamps, noise)
        self._show_circle(s)
        return s
    
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
    
class SymmetrySubgroup:
    
    def __init__(self, n):
        self.n = n
        self.theta = (2*np.pi)/n
        c,s = np.cos(self.theta), np.sin(self.theta)
        self.r = np.array([[c, -s],[s, c]])
        self.origin = self.r @ np.array([1, 0])
        
    def generate_polygon(self):
        vertices = np.zeros((self.n, 2))
        x = np.array([1, 0])
        r = self.r
        for n in range(self.n):
            x = r @ x 
            vertices[n,:] = x
        return vertices
    
    def show_polygon(self, show = True):
        vertices = self.generate_polygon()
        vertices_aug = np.vstack([vertices, vertices[0,:]])
        x, y = vertices.T
        plt.plot(vertices_aug[:,0], vertices_aug[:,1])
        plt.scatter(x, y, s = 20, color = 'black')
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.gca().set_aspect('equal')
        if show:
            plt.show()
            plt.close()
        return vertices
    
    def convolve(self, f, k):
        raise NotImplementedError
        

class Cn(SymmetrySubgroup):
    
    def __init__(self,n):
        super(Cn, self).__init__(n = n)
        
    def convolve(self, f, k):
        assert len(f) == len(k), 'Filter dim must match function dim.'
        assert len(f) == self.n, 'Function must be defined at each vertex.'
        
        h = np.zeros(self.n)
        for g in range(self.n):
            gk = np.roll(k, g)
            h[g] = (f*gk).sum()
        return h
    
    def f_action(self,g,f):
        return np.roll(f, g)
    
    def check_equivariance(self,f,k):
        x, y = self.generate_polygon().T
        h = self.convolve(f,k)
        plt.subplot(121)
        plt.scatter(x, y, s = 50, c = f)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Function')
        plt.colorbar()
        plt.gca().set_aspect('equal')
        
        plt.subplot(122)
        plt.scatter(x, y, s = 50, c = h, cmap = 'coolwarm')
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title('Covolution')
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()
        plt.close()
        
        for g in range(self.n):
            gf = self.f_action(g,f)
            hgf = self.convolve(gf,k)
            ghf = self.f_action(g,h)
            plt.subplot(131)
            plt.scatter(x, y, s = 50, c = gf)
            plt.xlim([-1.5, 1.5])
            plt.ylim([-1.5, 1.5])
            plt.title('Group-Function')
            #plt.colorbar()
            plt.gca().set_aspect('equal')
            plt.subplot(132)
            plt.scatter(x, y, s = 50, c = hgf, cmap = 'coolwarm')
            plt.xlim([-1.5, 1.5])
            plt.ylim([-1.5, 1.5])
            plt.title('Conv-Group-Function')
            #plt.colorbar()
            plt.gca().set_aspect('equal')
        
            plt.subplot(133)
            plt.scatter(x, y, s = 50, c = ghf, cmap = 'coolwarm')
            plt.xlim([-1.5, 1.5])
            plt.ylim([-1.5, 1.5])
            plt.title('Group-Conv-Function')
            #plt.colorbar()
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.show()
            plt.close()
            
def sinc_interp(x, s, u):
    '''
    adapted from https://gist.github.com/endolith/1297227
    '''
    if len(x) != len(s):
        raise ValueError('x and s must be the same length')
    
    # Find the period    
    T = s[1] - s[0]
    
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    y = np.dot(x, np.sinc(sincM/T))
    return y

def translate_with_inter(s, u, y, tau):
    s_trans = s + tau
    sig_trans = np.zeros(len(s))
    for i, s_val in enumerate(s_trans):
        idx = np.argmin((u - s_val)**2)
        sig_trans[i] = y[idx]
    return s_trans, sig_trans
