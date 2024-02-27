import numpy as np


class LogisticRegression:

    def __init__(self , n_iters , lr):
        self.n_iters = n_iters
        self.lr = lr
        self.z = None

    def sigmoid(self , z):
        return 1/(1 + np.exp(-z))
       
    def fit(self , X , y):
        num_samples , num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            model = np.dot(X , self.w) + self.b
            y_pred = self.sigmoid(model)     

            grad_w =  (1/num_samples) * np.dot(X.T , y_pred - y)
            grad_b = (1/num_samples) * np.sum(y_pred - y)
            
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            
    def predict(self , x):
        linear_model = np.dot(x, self.w) + self.b
        y_predicted = self.sigmoid(linear_model)
   
        y_predicted_cls = np.where(y_predicted > 0.5, 1, 0)
        return y_predicted_cls

            
  
X_train = np.array([[2, 3], [4, 5], [6, 7]])
y_train = np.array([0, 1, 1])

model = LogisticRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)

X_test = np.array([[1, 2], [5, 6]])
predictions = model.predict(X_test)
print(predictions)


