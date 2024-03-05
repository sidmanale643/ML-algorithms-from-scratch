import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self , dim):
        self.dim = dim

    def fit(self , X):
        
        self.mean = np.mean(X , axis = 0) #calculate mean of the features NOTE: here features are represented as rows unlike ML convention where they are columns
        X_std = X - self.mean  #subtract mean from X
        
        cov = np.cov(X_std.T) #calculate covariance of X , Transposed because the numpy funtion takes in values as columns
        print(cov.shape)
        #Eigenvectors are non-zero vectors that remain in the same direction after linear transformation
        #Eigenvalues are scalar values that represent the scaling factor of eigenvectors in a transformation
        eigen_vec , eigen_val = np.linalg.eig(cov) #calculate eigen vectors and eigen values                       
        # print(eigen_vec , eigen_val )                    
        idx = np.argsort(eigen_val)[::-1] #sort eigen vectors in descending order
        # print(idx)
        eigen_val = eigen_val[idx]                        
        eigen_vec = eigen_vec[idx]         
                     
        self.top_k_eigen_vec = eigen_vec[:self.dim] #choose the top 'k' eigen vectors
    
    def project(self , X): #projecting data
        
        X_std = X - np.mean(X)
        projection = np.dot(X_std , self.top_k_eigen_vec.T) #projection is just the dot product of the mean centred data and top k eigen vectors
        # print(f" top k eigen vectors : {top_k_eigen_vec}")                        
        return projection 
    

np.random.seed(50) 
X = np.random.randn(100, 4)  # 100 samples, 4 features

pca = PCA(2)  # Project the data onto the 2 principal components
pca.fit(X)
projection = pca.project(X)
print(projection)


x1 = projection[:, 0]
x2 =projection[:, 1]

plt.scatter(x1, x2, alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()