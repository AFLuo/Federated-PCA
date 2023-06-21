import numpy as np
from numpy import matrix as mt
np.random.seed(0)

def Eigen_PCA(X:np.array,var_cont):
    '''
        input: X, n by m, m is the number of features, n is the number of samples.
        This functions processes PCA with eigen_decomposition
    '''
    n,m=X.shape
    # mean-centering
    Y=X-X.mean(axis=0)
    # 排除量纲影响
    Y=Y/Y.std(axis=0)
    # compute covariance matrix
    Z=np.matmul(Y.T,Y)/(n-1)
    # Z is a square matrix m by m
    # compute eigendecomposition
    '''
        Two issues raises when calling np.linalg.eig:
            1. The eigenvalues returned are not real (i.e. in complex form)
            2. Some of the eigenvalues are negative
        Both of these issues are the result of errors introduced by truncation and rounding errors, 
            which always happen with iterative algorithms using floating-point arithmetic. 
        To avoid the first problem, call np.linalg.eigh, 
            an algorithm meant to handle a complex Hermitian (conjugate symmetric) or a real symmetric matrix, 
            both of which should noly contain real eigenvalues.
    '''
    eigenvalues,eigenvectors=np.linalg.eigh(Z)
    # sort the eigenvalues: descending order
    idx=eigenvalues.argsort()[::-1]
    eigenvalues=eigenvalues[idx]
    eigenvectors=eigenvectors[:,idx]
    # find the number of k which satisties the variance contribution rate
    k=0
    for i in range(m-1):
        if eigenvalues[:i+1].sum()/eigenvalues.sum() >= var_cont:
            k=i+1
            break
    # then unite the k biggest eigenvectors and everything is done
    for i in range(k):
        length=np.sqrt(np.dot(eigenvectors[:,i],eigenvectors[:,i]))
        eigenvectors[:,i]=eigenvectors[:,i]/length
    # you can possibly project existing sample matrix into new dimensions if you wish
    projected_X=np.matmul(X,eigenvectors[:,:k])
    return projected_X
    
        

def SVD_PCA(X:np.array):
    '''
        input: X, n by m, m is the number of features, n is the number of samples.
        This function processes PCA with Singular Value Decomposition solver.
        The number of Principal Components (k) is predefined.
    '''
    # construct new matrix
    n,m=X.shape
    # mean-cencering
    X=X-X.mean(axis=0)
    X_prime=X/(n-1)
    # truncated SVD: k singular values and k sigular vectors

    # projection
    pass

def randmized_SVD(X,k):
    '''
        Randomized SVD solver:
            Given an m by n matrix X, a target number k of singular vectors, and an
            exponent q, this procedure computes an approximate rank-2k factorization U \Delta V*, where U and V are orthonormal and \Delta is
            nonnegative and diagonal
    '''
    # pesudocodes: 
    # Stage A:
    #   Generate an n by 2k Gaussian test matrix \Omega
    #   Form Y=(AA*)^q A \Omega by multiplying alternatively with A and A*
    #   Construct a matrix Q whose columns form an orthnormal basis for the range of Y
    # Stage B:
    #   Form B=Q*A
    #   Compute an SVD of the small matrix B=\tilde U \Delta V*
    #   Set U=Q\tilde U
    pass
def randomized_PCA(X:np.array,k):
    '''
        input: X, n by m, m is the number of features, n is the number of samples.
        This function processes PCA with randomized SVD solver from Halko et al, which is discussed in SF-PCA.
    '''
    
    pass



def schmidt_orthon(input:np.matrix):
    pass

if __name__=="__main__":
    # randomly generate sample matrix
    X=np.random.normal(0,2,(5,10))
    print(Eigen_PCA(X,0.75))
