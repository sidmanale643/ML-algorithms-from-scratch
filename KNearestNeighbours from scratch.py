import numpy as np

class KNNRegression:
    def __init__(self , k):
        self.k = k
     
    def fit(self, X , y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
            
    def predict(self ,X):
        dist =  [np.sqrt(np.sum((X - x_train_pt)**2)) for x_train_pt in self.X_train]
        indices = np.argsort(dist)[:self.k]
        y = np.mean(self.y_train[indices])
        return y

# knn = KNNRegression(2)
# knn.fit( X = [ [2,3] , [6,3] ] , y=[4, 8] )
# print(knn.dist(X = [5,5]))

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()


reg = KNNRegression(k=5)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)