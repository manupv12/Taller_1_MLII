'''Here point 3 is implemented with the libraries that contain the SVD, PCA and TSNE methods'''

import numpy as np
from tsne import estimate_sne, q_tsne, q_joint, tsne_grad, symmetric_sne_grad, p_joint

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # center the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # compute the covariance matrix
        cov = np.cov(X, rowvar=False)

        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort the eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store the first n_components eigenvectors as the principal components
        self.components = eigenvectors[:, : self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        # center the data
        X = X - self.mean

        # project the data onto the principal components
        X_transformed = np.dot(X, self.components)

        return X_transformed

class SVD:
    def __init__(self, n_components=None):
        self.n_components = n_components
    
    def fit(self, X):
        X = np.array(X)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[:self.n_components].T if self.n_components else Vt.T
        self.explained_variance_ = (s ** 2) / (X.shape[0] - 1)
        self.singular_values_ = s
        # punto 4:
        self.U = U
        self.s = s
        self.Vt = Vt
        return U, s, Vt 
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X):
        X = np.array(X)
        return np.dot(X, self.components_)
    
    def inverse_transform(self, X, k=None):
        if k is None:
            k = self.n_components
        X = np.array(X)
        X_transformed = np.dot(X, self.components_[:k, :].T)
        return X_transformed

    

class TSNE:
    def __init__(self, n_components=2, perplexity=12.0, learning_rate=12, n_iter=1000, seed = 42, momentum=0.9):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.seed = seed
        self.tse = True
        self.momentum = momentum
        
    def fit(self, X):
        # Implementation of the T-SNE algorithm
        pass
    
    def fit_transform(self, X):
        """
        Y = estimate_sne(X, y, P, rng,
             num_iters=NUM_ITERS,
             q_fn=q_tsne if TSNE else q_joint,
             grad_fn =tsne_grad if TSNE else symmetric_sne_grad,
             learning_rate=LEARNING_RATE,
             momentum=MOMENTUM,
             plot=NUM_PLOTS)
        """
        P = p_joint(X, self.perplexity)
        rng = np.random.RandomState(self.seed)
        Y = estimate_sne(X, P, rng, self.n_iter,
                        q_fn= q_tsne if  self.tse else q_joint,
                        grad_fn=tsne_grad if self.tse else symmetric_sne_grad,
                        learning_rate=self.learning_rate,
                        momentum=self.momentum )
        return Y
        
    def transform(self, X):
        # Implement transformation of new data based on the existing model
        pass