import numpy as np #import the numpy library which is used for array and matrix manipulation


class LogisticRegression: #create a Logistic Regression Class

    def __init__(self , n_iters , lr): #constructor that automatically initializes the values of number of iterations and learning rate when a class object is created 
        self.n_iters = n_iters 
        self.lr = lr
        self.z = None

    def sigmoid(self , z): #The sigmoid function is a mathematical function that maps input values to a range between 0 and 1 ; z = σ(x) = 1/(1+e^−x)
​        return 1/(1 + np.exp(-z))
       
    def fit(self , X , y): #fit the model to training data
        num_samples , num_features = X.shape 
        self.w = np.zeros(num_features) #initialize the weights as zeros with the same dimension as the number of features
        self.b = 0 #initialize the bias as zero
        
        for _ in range(self.n_iters): #number of convergence steps is equal to the specified number of iterations 
            model = np.dot(X , self.w) + self.b #linear model ; z = X*w + b
            y_pred = self.sigmoid(model) #applying the sigmoid function to the linear model to get values ranging between 0 and 1     

            grad_w =  (1/num_samples) * np.dot(X.T , y_pred - y)  #gradient equation of the bias with respect to the error (predicted_value - actual_value)  
            grad_b = (1/num_samples) * np.sum(y_pred - y)  #gradient equation of the bias with respect to the error (predicted_value - actual_value)  

            
            self.w -= self.lr * grad_w #update the weight parameters according to the gradient 
            self.b -= self.lr * grad_b #update the bias term with according to the bias
            
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


