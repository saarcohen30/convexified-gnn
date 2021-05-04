"""
This code implements a single convexified graph (convolutional) filter,
which corresponds to a single layer of a Convexified GNN (Cx-GNN).
It is largely inspired by the following paper:

[Zhang et al., 2017]
Yuchen Zhang, Percy Liang, and Martin J Wainwright.   
Convexified convolutional neural networks. 
In International Conference on Machine Learning, 
pages 4044-4053. PMLR, 2017.

However, the original code was not suitable for our own context, 
and was thus vastly altered in a manner that it would be capable 
of processing graph signals. We note that instead of a regular 
projected gradient decsent, we employ an ADAM optimizer, where
the optimized parameters are then projected onto the nuclear
norm ball. Additionally, instead of NumPy Pytorch was utilized 
for the sake of the implementation.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_
from torch.nn.modules.module import Module

import numpy as np
from numpy import linalg as LA
import random
import math, sys
import sklearn
import numexpr as ne
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
from sklearn.utils.extmath import safe_sparse_dot
import datetime
import time

np.set_printoptions(precision=4, suppress=True, threshold=1000, linewidth=500)
ne.set_num_threads(32)
random.seed(1)



def tprint(s):
    """ 
    An enhanced print function with time concatenated to the output.
    Source: Convexified Convolutional Neural Networks, by Zhang et al.
    """
    tm_str = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    print(tm_str + ":  " + str(s))
    sys.stdout.flush()

class NystroemTransformer:
    reference_matrix = 0
    transform_matrix = 0
    n_components = 0
    gamma = 0

    def __init__(self, gamma, n_components, device):
        self.n_components = n_components
        self.gamma = gamma
        self.device = device

    def fit(self, X):
        n = X.shape[0]
        index = torch.randint(0, n, (self.n_components,)).to(self.device)
        self.reference_matrix = X[index].clone()
        kernel_matrix = rbf_kernel_matrix(gamma=self.gamma, X=self.reference_matrix, Y=self.reference_matrix).to(self.device)
        (U, s, V) = torch.svd(kernel_matrix)
        # self.transform_matrix = torch.matmul(U, torch.matmul(torch.diag_embed(1.0/(torch.sqrt(s + 0.0001e-04))), V)).to(self.device)
        
        S_nonzero = s.clone()
        # S_nonzero[torch.nonzero(S_nonzero <= 1e-12, as_tuple=True)] = 1e-12
        S_nonzero[torch.nonzero(S_nonzero <= 1e-12)] = 1e-12
        self.transform_matrix = torch.matmul(U, torch.matmul(torch.diag_embed(1.0/(torch.sqrt(S_nonzero))), V)).to(self.device)

    def transform(self, Y):
        kernel_matrix = rbf_kernel_matrix(gamma=self.gamma, X=self.reference_matrix, Y=Y).to(self.device)
        output = (torch.matmul(self.transform_matrix, kernel_matrix)).transpose(1, 0).to(self.device)
        
        return output
        
def rbf_kernel_matrix(gamma, X, Y):
    nx = X.shape[0]
    ny = Y.shape[0]    

    X2 = torch.matmul(torch.sum(X**2, dim=1).reshape((nx, 1)), torch.FloatTensor(torch.ones(1,ny)).to(X.device))
    Y2 = torch.matmul(torch.FloatTensor(torch.ones(nx,1)).to(X.device), torch.sum(Y**2, dim=1).reshape((1, ny)))    
    # XY = torch.matmul(X, Y.transpose(1, 0))
    
    distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
    distances += X2
    distances += Y2    
    
    distances = torch.maximum(distances, torch.zeros(distances.shape).to(distances.device))
    
    # exp(gamma*(2*XY-X2-Y2))
    # out = torch.exp(gamma * (2 * XY - X2 - Y2))
    # # return torch.exp(gamma * (2 * XY - X2 - Y2))
    # if torch.sum(torch.isinf(out)) > 0:
    #     """
    #     We apply normalization, in regard with scenarios in which the features' values are too high, yielding Inf values
    #     after their multiplication.
    #     """
    #     X_norm = X - X.min(1, keepdim=True)[0]
    #     X_norm /= X_norm.max(1, keepdim=True)[0]
    #     Y_norm = Y - Y.min(1, keepdim=True)[0]
    #     Y_norm /= Y_norm.max(1, keepdim=True)[0]
    #     X2 = torch.matmul(torch.sum(X_norm**2, dim=1).reshape((nx, 1)), torch.FloatTensor(torch.ones(1,ny)).to(X.device))
    #     Y2 = torch.matmul(torch.FloatTensor(torch.ones(nx,1)).to(X.device), torch.sum(Y_norm**2, dim=1).reshape((1, ny)))    
        # XY = torch.matmul(X_norm, Y_norm.transpose(1, 0))
        
        # return torch.exp(gamma * (2 * XY - X2 - Y2))
    return torch.exp(-gamma * distances)
    

def project_to_nuclear_norm(A, R, P, nystrom_dim, d2):
    """
    Dependencies: euclidean_proj_simplex, euclidean_proj_l1ball
    
    Parameters
    ----------
    @param A:  matrix to be projected onto the nuclear norm ball
    @param R: upper bound of nuclear norm.
    @param P: number of patches
    @param nystroem_dim: Nystroem dimension. (p)
    @param d2: Number of classes for the categorical classification
    
    -------    
    Returns
    -------
    @return Ahat: numpy array,
          Projection of A on the nuclear norm ||A||_{*} = R
    @return U, s, V: A's singular vectors and values
    """
    # print("A " + str(A.shape))
    A = A.reshape(d2, P*nystrom_dim) #.cuda()
    # A = np.reshape(A, ((n_classes-1)*P, nystroem_dim))
    (U, s, V) = torch.svd(A)
    S_nonzero = s.clone()
    # S_nonzero[torch.nonzero(S_nonzero <= 1e-12, as_tuple=True)] = 1e-12
    S_nonzero[torch.nonzero(S_nonzero <= 1e-12)] = 1e-12
    s = project_onto_l1_ball(S_nonzero, eps=R)
    Ahat = (torch.matmul(torch.matmul(U, torch.diag_embed(s)), V.transpose(1, 0))).reshape((d2, P*nystrom_dim))
    return Ahat, U, s, V

def project_onto_l1_ball(x, eps=1):
    """
    Compute Euclidean projection onto the L1 ball for a batch.
    
      min ||x - u||_2 s.t. ||u||_1 <= eps
    
    Inspired by the corresponding numpy version by Adrien Gaidon.
    
    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU
      
    eps: float
      radius of l-1 ball to project onto
    
    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original
    
    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.
    
    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    x_norm = x - torch.mean(x, dim=0)
    x_norm = x_norm / torch.norm(x_norm)
    mask = (torch.norm(x_norm, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x_norm), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1).float()
    arange = torch.arange(1, x_norm.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange.float() > (cumsum - eps)).float() * arange.float(), dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), (rho.cpu() - 1).long()] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x_norm + (1 - mask) * proj * torch.sign(x_norm)
    return x.view(original_shape)


class CGraphFilter(nn.Conv2d):
# class CGraphFilter(Module):
    """ 
    Function class for generating a Convexified Convolutional Neural Network.
    
    Functions:
        __init__: initializes the convexified graph filter class. 
        construct_Q: Constructs Q by approximating the kernel matrix K.
        train: Trains the convexified graph filter with respect to Q and Y.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 device,
                 is_last_layer=False,
                 nystrom_dim=20, 
                 gamma=0.2, 
                 R=1, 
                 learning_rate=0.2, 
                 n_iter=5000, 
                 print_iterations=250):
        """ 
        Initializes the convexified graph filter model.
        
        ----------
        Parameters
        ----------
        @param device: CUDA device for torch
        @param nystrom_dim: Nystrom dimension used for approximating the kernel matrix 
                     (corresponds to the parameter p).
        @param gamma: hyperparameter for the Gaussian RBF kernel. 
        @param R: Utilized for the sake of Euclidean projection onto a nuclear norm ball with radius R.
        @param n_iter: Number of iterations for the Projected Stochastic Gradient Descent.
        @param print_interval: How often to print current loss and accuracy to the console. 
                        1 means it printing at each iteration.
        """
        # super(CGraphFilter, self).__init__()
        super(CGraphFilter, self).__init__(in_channels, out_channels, kernel_size=(stride, 1), stride=(stride, 1))
        
        # Storing data properties
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.cnn = cnn
        # xavier_uniform_(self.weight.data)
        # xavier_normal_(self.bias.data)
        # self.weight.requires_grad_(True)
        # self.bias.requires_grad_(True)
        
        # Storing hyperparameters
        self.nystrom_dim = nystrom_dim # p
        self.gamma = gamma 
        self.R = R
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.print_iterations = print_iterations
        self.step = stride
        self.is_last_layer = is_last_layer
        self.device = device
        self.A_sum = torch.zeros(self.weight.shape).to(self.device)
        
        # self.project()
    
    def construct_Q(self, feature_normalization=True):
        """
        Computes Q, such that K ~ QQ^T constitutes a suitable approximation for the 
        Gaussian RBF kernel matrix K. Normalization to the features is also applied 
        by default. This is carried out in regard with Zhang et al.'s implementation 
        for the convexified graph filter architecture.
            
        ----------
        Parameters
        ----------
        @params Z: (N,P,d1) arrays. Each Z[i,:,:] is one Z(x_i). 
                         Result from __init__.
        
        ------
        Return
        ------
        @return Q: (N,P,m) arrays Each Q[i,:,:] is one Q(x_i). 
                         Used in train() function below.
        """    
        
        # tprint("Using Scikitlearn Nystroem function")
        # tprint("Creating Q...")
        # if torch.sum(self.Z == 0) >= 1200:
        #     Z_shape = self.Z.shape
        #     prod = Z_shape[0]
        #     for i in range(1, len(Z_shape)):
        #         prod *= Z_shape[i]
        #     Z_sample = self.Z.reshape(prod)
        #     for i in range(len(Z_sample)):
        #         if Z_sample[i] == 0:
        #             Z_sample[i] = 0.0001
        #     self.Z = Z_sample.reshape(Z_shape)
            
        transformer = NystroemTransformer(gamma=self.gamma, n_components=self.nystrom_dim, device=self.device)
        transformer.fit(X=self.Z)
        self.Q = transformer.transform(Y=self.Z)

        # Z_train = self.Z.clone().cpu().detach().numpy()
        # transformer = Nystroem(gamma=self.gamma, n_components=self.nystrom_dim)
        # transformer = transformer.fit(X=Z_train)
        # self.Q = torch.FloatTensor(transformer.transform(Z_train)).to(self.device)
        
        with torch.no_grad():
            if feature_normalization == True:
                # self.Q = self.normalize(self.Q)
                self.Q.sub_(torch.mean(self.Q, dim=0))
                self.Q.div_(torch.norm(self.Q) / math.sqrt(self.b*self.n) + 1e-8)

    def project(self):
        """
        During training, the actor's parameters correspond to all the filters it is 
        comprised of. Hence, each convexified graph filter shall be trained via 
        Projected Stochastic Gradient Descent (or, alternately, via an ADAM optimizier) 
        in regard with each batch of training samples. That is, after optimization,
        projection to the nuclear norm shall be applied to the learned parameters matrix. 
        """
        
        nystrom_dim = self.step * self.in_channels
        
        A_avg, U, s, V = project_to_nuclear_norm(A=self.weight, 
                                                      R=self.R, 
                                                      P=1, 
                                                      nystrom_dim=nystrom_dim,
                                                      d2=self.out_channels)
        
        A_avg = A_avg.reshape(self.A_sum.shape)
        self.A_sum = A_avg.to(self.device)
                
        # if self.is_last_layer:
        #     self.U = V
        # else:
        #     self.U = U
        self.U = U
        self.U = self.U.reshape((self.out_channels, self.in_channels , self.step, 1)).to(self.device)

    def forward(self, x, is_test): 
        """
        Applies the filters, learned via Projected Stochastic Gradient Descent,
        on the graph signal.
        
        Requires:
        -----
        self.weight: The learned filters.

        Parameters
        ----------
        n_iter: number of iterations for the Projected Stochastic Gradient Descent
        print_interval: how often to print current loss and accuracy to 
                          the console. 1 means it prints each iteration.        
        """        
        if is_test:
            return F.conv2d(x, self.A_sum, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
                            
        self.b, self.f, self.k, self.n = x.shape
        self.nystrom_dim = self.k * self.f
        # print("x pre " + str(x.shape))
        self.Z = x.reshape((self.b * self.n, self.k * self.f))
        # self.Z = self.normalize(self.Z)
        # with torch.no_grad():
        #     self.Z = self.Z.cpu().numpy()
        
        # Calculating a suitable approximation for the Gaussian RBF kernel matrix K
        self.construct_Q()
        self.Q = self.Q.reshape((self.b, self.f, self.k, self.n))
        # print("Q " + str(self.Q.shape))
        
        # out = F.conv2d(self.Q, self.U, self.bias, self.stride,
        #                 self.padding, self.dilation, self.groups)
        
        return F.conv2d(self.Q, self.U, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        # out_norm = out.clone()
        # out_norm.sub_(torch.mean(out_norm, axis=0))
        # out_norm.div_(torch.norm(out_norm) / math.sqrt(self.b*self.n))

        # return out_norm
                
            
            