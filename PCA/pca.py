import numpy as np
from numpy import matrix as mt
np.random.seed(0)

def randomized_pca(input:np.array):
    pass
def classic_PCA(X:np.array,var_cont):
    '''
        input: X, n by m, m is the number of features, n is the number of samples.
        This functions processes PCA with eigen_decomposition
    '''
    # mean-centering
    Y=X-X.mean(axis=0)
    # 排除量纲影响
    Y=Y/Y.std(axis=0)
    # compute covariance matrix
    Z=np.matmul(Y.T,Y)
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
    for i in range(X.shape[1]):
        if eigenvalues[:i+1].sum()/eigenvalues.sum() >= var_cont:
            k=i+1
            break
    
    
    # 

def modern_PCA(X:np.array):
    '''
        input: X, n by m, m is the number of features, n is the number of samples.
        This function processes PCA with Singular
    '''
    pass

def randomized_PCA(X:np.array):
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
    classic_PCA(X)
